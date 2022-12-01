from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from ms2ml.config import Config
from ms2ml.data.adapters.base import BaseAdapter
from ms2ml.data.adapters.mzml import MZMLAdapter
from ms2ml.data.adapters.pin import _find_raw_file
from ms2ml.data.parsing.mokapot import MokapotPSMParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum
from ms2ml.type_defs import PathLike


class MokapotPSMAdapter(BaseAdapter, MokapotPSMParser):
    def __init__(
        self,
        file: PathLike,
        config: Config,
        in_hook: Callable | None = None,
        out_hook: Callable | None = None,
        collate_fn: Callable[..., Any] | None = None,
        raw_file_locations: list[PathLike] | PathLike = ".",
    ):
        BaseAdapter.__init__(
            self,
            config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        MokapotPSMParser.__init__(self, file)

        self.mzml_adapters = {}

        if raw_file_locations is None:
            self.raw_file_locations = ["."]
        elif isinstance(raw_file_locations, (str, Path)):
            self.raw_file_locations = [raw_file_locations]
        else:
            self.raw_file_locations = raw_file_locations

    def parse_file(self, file: PathLike) -> Iterator[AnnotatedPeptideSpectrum]:
        """Parses a pin file and yields one spectrum at a time.

        The spectra are yielded as AnnotatedPeptideSpectrum"""
        if file != self.file:
            adapter = MokapotPSMAdapter(
                file, self.config, self.in_hook, self.out_hook, self.collate_fn
            )
            yield from adapter.parse_file(file)
        else:
            for spec in super().parse_file(file):
                yield self._process_elem(spec)

    def _to_elem(self, spec_dict: dict) -> AnnotatedPeptideSpectrum:
        """
        {'CalcMass': 1486.728483,
        'ExpMass': 1486.729129,
        'Label': True,
        'MatchRank': 1,
        'NextAminoAcid': 'K',
        'Peptide': 'R.HRLDLGEDYPSGK.K',
        'PeptideSequence': 'HRLDLGEDYPSGK',
        'PrecursorCharge': 3,
        'PreviousAminoAcid': 'R',
        'Proteins': 'sp|O43143|DHX15_HUMAN',
        'RawFile': 'sample_tiny_hela',
        'ScanNr': 10044,
        'SpecId': 'sample_tiny_hela_10044_3_1',
        'SpectrumIndex': 10044,
        'mokapot PEP': 1.9301281018285933e-07,
        'mokapot q-value': 0.0714285714285714,
        'mokapot score': 3.364551}

        """
        seq = f"{spec_dict['peptidesequence']}/{spec_dict['precursorcharge']}"
        pep = Peptide.from_proforma_seq(seq, config=self.config)

        if spec_dict["rawfile"] not in self.mzml_adapters:
            self.mzml_adapters[spec_dict["rawfile"]] = MZMLAdapter(
                _find_raw_file(self.raw_file_locations, spec_dict["rawfile"]),
                self.config,
            )

        spec = self.mzml_adapters[spec_dict["rawfile"]][spec_dict["spectrumindex"]]
        spec = spec.annotate(pep)
        spec.extras.update(spec_dict)

        return spec
