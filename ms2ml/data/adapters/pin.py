from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterator

from ms2ml.config import Config
from ms2ml.data.parsing.pin import PinParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter
from .mzml import MZMLAdapter


class PinAdapter(PinParser, BaseAdapter):
    def __init__(
        self,
        file: PathLike[Any],
        config: Config,
        in_hook: Callable = None,
        out_hook: Callable = None,
        collate_fn: Callable[..., Any] = None,
        raw_file_locations: list[PathLike] = None,
    ):
        BaseAdapter.__init__(
            self,
            config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        PinParser.__init__(self, file)
        self.mzml_adapters = {}

        if raw_file_locations is None:
            self.raw_file_locations = ["."]
        elif isinstance(raw_file_locations, (str, Path, PathLike)):
            self.raw_file_locations = [raw_file_locations]
        else:
            self.raw_file_locations = raw_file_locations

    def parse_file(self, file) -> Iterator[AnnotatedPeptideSpectrum]:
        for spec in super().parse_file(file):
            yield self._process_elem(spec)

    def _to_elem(self, spec_dict) -> AnnotatedPeptideSpectrum:
        seq = f"{spec_dict['PeptideSequence']}/{spec_dict['PrecursorCharge']}"
        pep = Peptide.from_proforma_seq(seq, config=self.config)

        if spec_dict["RawFile"] not in self.mzml_adapters:
            self.mzml_adapters[spec_dict["RawFile"]] = MZMLAdapter(
                self.find_raw_file(spec_dict["RawFile"]), self.config
            )

        spec = self.mzml_adapters[spec_dict["RawFile"]][spec_dict["SpectrumIndex"]]
        spec = spec.annotate(pep)
        spec.extras.update(spec_dict)

        return spec

    def _process_elem(self, elem: AnnotatedPeptideSpectrum) -> AnnotatedPeptideSpectrum:
        elem = super()._process_elem(elem)
        return elem

    def find_raw_file(self, raw_file: str) -> str:
        outs = []
        for loc in self.raw_file_locations:
            tmp = list(Path(loc).rglob(f"*{raw_file}*"))
            outs.extend(tmp)

        outs = [x.resolve() for x in outs if "mzML" in str(x)]
        outs = list(set(outs))
        if len(outs) == 0:
            raise FileNotFoundError(f"Could not find {raw_file}.mzML")
        elif len(outs) > 1:
            raise FileNotFoundError(f"Found multiple files for {raw_file}.mzML, {outs}")

        return outs[0]
