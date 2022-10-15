import re
from typing import Callable, Optional

from pyteomics.mzml import read as read_mzml

from ms2ml.annotation_classes import RetentionTime
from ms2ml.config import Config
from ms2ml.data.adapters import BaseAdapter
from ms2ml.data.utils import pad_collate
from ms2ml.spectrum import Spectrum


class MZMLAdapter(BaseAdapter):
    def __init__(
        self,
        file,
        config: Config,
        out_hook: Optional[Callable] = None,
        in_hook: Optional[Callable] = None,
        collate_fn: Callable = pad_collate,
    ):
        """Provides an adapter for mzML files.

        Args:
            file (str): Path to the file to be parsed.
            config (Config): The config object to use.
            out_hook (Optional[Callable], optional): A function to apply to the
                output of the adapter. Defaults to None.
            in_hook (Optional[Callable], optional): A function to apply to the
                input of the adapter. Defaults to None.
            collate_fn (Callable, optional): A function to use to collate the
                output of the adapter. Defaults to pad_collate.
        """
        super().__init__(
            config=config,
            out_hook=out_hook,
            in_hook=in_hook,
            collate_fn=collate_fn,
        )
        self.file = str(file)
        self.config = config
        self.out_hook = out_hook

        with read_mzml(self.file) as reader:
            # controllerType=0 controllerNumber=1 scan=2634
            self._index_example = next(reader)["id"]
            if "scan" in self._index_example:
                template = re.sub(r"(?<=scan\=)\d+", "{}", self._index_example)
                self._index_template = template

    def parse(self):
        with read_mzml(self.file) as reader:
            for spec in reader:
                yield self._process_elem(spec)

    def _to_elem(self, spec_dict):
        if "centroid spectrum" not in spec_dict:
            raise NotImplementedError(
                "Only centroid spectra are supported at the moment."
            )

        if "MSn spectrum" in spec_dict:
            if spec_dict["precursorList"]["count"] != 1:
                raise NotImplementedError
            precursor = spec_dict["precursorList"].pop("precursor")
            precursor = precursor[0]
            precursor["spectrumRef"]

            if precursor["selectedIonList"]["count"] != 1:
                raise NotImplementedError

            selected_ion = precursor["selectedIonList"].pop("selectedIon")
            selected_ion = selected_ion[0]

            precursor_mz = selected_ion["selected ion m/z"]
            _ = selected_ion["peak intensity"]
            precursor_charge = selected_ion["charge state"]

            precursor.update(selected_ion)
            precursor_extras = {
                ("precursor " if not k.startswith("precursor") else "") + k: v
                for k, v in precursor.items()
            }
            spec_dict.update(precursor_extras)

        elif "MS1 spectrum" in spec_dict:
            precursor_mz = None
            precursor_charge = None

        rt = min([x["scan start time"] for x in spec_dict["scanList"]["scan"]])
        rt = RetentionTime(rt=float(rt), units=rt.unit_info, run=self.file)

        mz = spec_dict.pop("m/z array")
        intensity = spec_dict.pop("intensity array")
        ms_level = spec_dict.pop("ms level")

        # TODO figure out how to get the analyzer
        spec_out = Spectrum(
            mz=mz,
            intensity=intensity,
            ms_level=ms_level,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            analyzer=None,
            retention_time=rt,
            extras=spec_dict,
            config=self.config,
        )
        spec_out.base_peak = spec_dict["base peak intensity"]
        spec_out.tic = spec_dict["total ion current"]

        return spec_out

    def __repr__(self):
        return f"MS2ML MZML Adapter for {self.file}"

    def __getitem__(self, idx):
        with read_mzml(self.file) as reader:
            try:
                spec = reader[idx]
            except TypeError:
                spec = None
                _ = "TypeError: 'NoneType' object is not subscriptable"
                err = f"Unable to find index {idx} in {self.file},"
                err += f" an example index is '{self._index_example}'"
                err = IndexError(err)

                if hasattr(self, "_index_template"):
                    try:
                        spec = reader[self._index_template.format(idx)]
                    except TypeError:
                        pass

                if spec is None:
                    raise err

        out = self._process_elem(spec)
        return out
