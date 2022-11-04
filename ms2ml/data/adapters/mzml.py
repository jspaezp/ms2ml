import re
from functools import lru_cache
from typing import Callable, Generator, Optional

import pandas as pd
from loguru import logger
from pyteomics import auxiliary as aux
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
        logger.debug(f"Loading mzML file: {file}")
        super().__init__(
            config=config,
            out_hook=out_hook,
            in_hook=in_hook,
            collate_fn=collate_fn,
        )
        self.file = str(file)
        self.config = config
        self.out_hook = out_hook
        self.reader = read_mzml(self.file, use_index=True, huge_tree=True)

        # controllerType=0 controllerNumber=1 scan=2634
        self._index_example = next(self.reader)["id"]
        if "scan" in self._index_example:
            template = re.sub(r"(?<=scan\=)\d+", "{}", self._index_example)
            self._index_template = template

        self.reader.reset()

        self.reader.iterfind('referenceableParamGroup[@id="CommonInstrumentParams"]')
        instrument_data: dict
        instrument_data = aux.cvquery(next(self.reader))  # type: ignore[operator]
        self.instrument = list(instrument_data.values())[0]
        logger.debug(f"Done loading mzML file: {file}")

    def parse(self) -> Generator[Spectrum, None, None]:
        """Parse the file and yield the spectra.

        Raises:
            NotImplementedError: If the file contains a spectrum that is not
                centroided.

        Examples:
            > adapter = MZMLAdapter("tests/data/BSA1.mzML")
            > for spec in adapter.parse():
            >     print(spec)

        """
        self.reader.reset()
        for spec in self.reader:
            spec["instrument"] = self.instrument
            yield self._process_elem(spec)

    def parse_ms1(self) -> Generator[Spectrum, None, None]:
        """Parse the file and yield the MS1 spectra.

        Examples:
            > adapter = MZMLAdapter("tests/data/BSA1.mzML")
            > for spec in adapter.parse_ms1():
            >     print(spec)
        """
        self.reader.reset()
        for spec in self.reader:
            if "MS1 spectrum" in spec:
                yield self._process_elem(spec)

    def get_chromatogram(self) -> pd.DataFrame:
        main_out = []
        for spec in self.parse():
            out = {}
            out["RTinSeconds"] = spec.retention_time.seconds()
            out["base_peak"] = spec.base_peak
            out["tic"] = spec.tic
            out["spec_id"] = spec.extras["id"]
            out["ms_level"] = spec.ms_level
            out["iso_window"] = spec.extras.get("IsolationWindow", None)
            main_out.append(out)

        return pd.DataFrame(main_out)

    def _to_elem(self, spec_dict: dict) -> Spectrum:
        if "centroid spectrum" not in spec_dict:
            raise NotImplementedError(
                "Only centroid spectra are supported at the moment."
            )

        if "MSn spectrum" in spec_dict:
            if spec_dict["precursorList"]["count"] != 1:
                raise NotImplementedError
            precursor = spec_dict["precursorList"].pop("precursor")
            precursor = precursor[0]

            prec_scan = precursor["spectrumRef"]
            precursor["ScanNumber"] = prec_scan[prec_scan.index("scan=") + 5 :]

            iso_window = precursor["isolationWindow"]

            iso_range = (
                iso_window["isolation window target m/z"]
                - iso_window["isolation window lower offset"],
                iso_window["isolation window target m/z"]
                + iso_window["isolation window upper offset"],
            )
            spec_dict["IsolationWindow"] = iso_range

            if precursor["selectedIonList"]["count"] != 1:
                raise NotImplementedError

            selected_ion = precursor["selectedIonList"].pop("selectedIon")
            selected_ion = selected_ion[0]

            precursor_mz = selected_ion["selected ion m/z"]

            # ocasionally the peak intensity will be missing, I am not sure why
            # _ = selected_ion["peak intensity"]
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

        else:
            msg = f"Only MS1 and MSn spectra supported. got: {spec_dict}"
            raise NotImplementedError(msg)

        rt = min([x["scan start time"] for x in spec_dict["scanList"]["scan"]])
        rt = RetentionTime(rt=float(rt), units=rt.unit_info, run=self.file)

        mz = spec_dict.pop("m/z array")
        intensity = spec_dict.pop("intensity array")
        ms_level = spec_dict.pop("ms level")
        instrument = spec_dict.pop("instrument")

        # TODO figure out how to get the analyzer
        spec_out = Spectrum(
            mz=mz,
            intensity=intensity,
            ms_level=ms_level,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            analyzer=None,
            instrument=instrument,
            retention_time=rt,
            extras=spec_dict,
            config=self.config,
        )
        spec_out.base_peak = spec_dict["base peak intensity"]
        spec_out.tic = spec_dict["total ion current"]

        return spec_out

    def __repr__(self):
        return f"MS2ML MZML Adapter for {self.file}"

    @lru_cache(maxsize=10)
    def __getitem__(self, idx):
        reader = self.reader

        spec = None
        if hasattr(self, "_index_template"):
            try:
                spec = reader[self._index_template.format(idx)]
            except TypeError:
                pass

        if spec is None:
            try:
                spec = reader[idx]
            except TypeError:
                _ = "TypeError: 'NoneType' object is not subscriptable"
                err = f"Unable to find index {idx} in {self.file},"
                err += f" an example index is '{self._index_example}'"
                err = IndexError(err)
                raise err

        spec["instrument"] = self.instrument  # type: ignore
        out = self._process_elem(spec)
        return out
