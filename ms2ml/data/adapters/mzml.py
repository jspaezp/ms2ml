import re
from functools import cached_property, lru_cache
from typing import Callable, Generator, Optional

import pandas as pd
from loguru import logger
from pyteomics import auxiliary as aux
from pyteomics.mzml import read as read_mzml

from ms2ml.annotation_classes import RetentionTime
from ms2ml.config import Config
from ms2ml.data.adapters import BaseAdapter
from ms2ml.spectrum import Spectrum
from ms2ml.utils.tensor_utils import pad_collate


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
        self._index_ids = list(self.reader.index.mapping["spectrum"])
        self._index_example = self._index_ids[0]

        if "scan" in self._index_example:
            self._index_template = "(.*scan)={SCAN_NUM}(\\s|$)"
        elif "index" in self._index_example:
            self._index_template = "(.*index)={SCAN_NUM}(\\s|$)"
        else:
            raise RuntimeError("Cound not determine scan names for the file.")

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

    def get_scan_info(self) -> pd.DataFrame:
        main_out = []
        for spec in self.parse():
            out = {}
            out["RTinSeconds"] = spec.retention_time.seconds()
            out["base_peak"] = spec.base_peak
            out["tic"] = spec.tic
            out["num_peaks"] = len(spec.mz)
            out["spec_id"] = spec.extras["id"]
            out["ms_level"] = spec.ms_level
            out["iso_window"] = spec.extras.get("IsolationWindow", None)
            out["collision_energy"] = spec.collision_energy
            main_out.append(out)

        return pd.DataFrame(main_out)

    def _to_elem(self, spec_dict: dict) -> Spectrum:
        if "centroid spectrum" not in spec_dict:
            raise NotImplementedError(
                "Only centroid spectra are supported at the moment."
            )

        if ("MSn spectrum" in spec_dict) or (
            "ms level" in spec_dict and spec_dict["ms level"] > 1
        ):
            if spec_dict["precursorList"]["count"] != 1:
                raise NotImplementedError
            precursor = spec_dict["precursorList"].pop("precursor")
            precursor = precursor[0]

            activation = precursor.pop("activation")
            activation_type = list(activation.values())[0]
            activation_energy = (
                activation["collision energy"]
                if "collision energy" in activation
                else float("nan")
            )

            if "spectrumRef" in precursor:
                prec_scan = precursor["spectrumRef"]
                precursor["ScanNumber"] = re.match(
                    self._index_template.format(SCAN_NUM="(.+)"), prec_scan
                ).groups(1)
            else:
                precursor["ScanNumber"] = float("nan")

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

            # ocasionally the peak intensity will be missing, I am not sure why ...
            # ATM I thit it happens when no peak was seelcted specifically,
            # but DIA scans contain the field with intensity of 0
            # _ = selected_ion["peac intensity"]
            if "charge state" in selected_ion:
                precursor_charge = selected_ion["charge state"]
            else:
                # DIA scans do not have precursor charge
                precursor_charge = None

            precursor.update(selected_ion)
            precursor_extras = {
                ("precursor " if not k.startswith("precursor") else "") + k: v
                for k, v in precursor.items()
            }
            spec_dict.update(precursor_extras)

        elif ("MS1 spectrum" in spec_dict) or (
            "ms level" in spec_dict and spec_dict["ms level"] == 1
        ):
            precursor_mz = None
            precursor_charge = None
            activation_type = None
            activation_energy = float("nan")

        else:
            msg = f"Only MS1 and MSn spectra supported. got: {spec_dict}"
            raise NotImplementedError(msg)

        rt = min([x["scan start time"] for x in spec_dict["scanList"]["scan"]])
        rt = RetentionTime(rt=float(rt), units=rt.unit_info, run=self.file)
        ims = spec_dict.get("precursor inverse reduced ion mobility", None)

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
            activation=activation_type,
            collision_energy=activation_energy,
            instrument=instrument,
            retention_time=rt,
            extras=spec_dict,
            precursor_ion_mobility=ims,
            config=self.config,
        )
        spec_out.base_peak = (
            spec_dict["base peak intensity"]
            if "base peak intensity" in spec_dict
            else intensity.max()
        )
        spec_out.tic = (
            spec_dict["total ion current"]
            if "total ion current" in spec_dict
            else intensity.sum()
        )

        return spec_out

    def __repr__(self):
        return f"MS2ML MZML Adapter for {self.file}"

    @cached_property
    def num_to_scan_ids(self):
        scannum_regex = re.compile(self._index_template.format(SCAN_NUM=r"(\d+)"))
        keep_dict = {}

        for x in self._index_ids:
            _key = int(scannum_regex.match(x).group(2))
            if _key in keep_dict:
                raise RuntimeError(f"Duplicate scan number {_key} found in {self.file}")
            keep_dict[_key] = x

        return keep_dict

    @lru_cache(maxsize=20)
    def __getitem__(self, idx):
        reader = self.reader
        spec = None

        if hasattr(self, "_index_template") and isinstance(idx, int):
            index_string = self.num_to_scan_ids[idx]
            spec = reader[index_string]

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
