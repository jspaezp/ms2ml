from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from loguru import logger

from ms2ml.config import Config
from ms2ml.data.adapters.base import BaseAdapter
from ms2ml.data.adapters.mzml import MZMLAdapter
from ms2ml.data.adapters.pin import _find_raw_file
from ms2ml.data.parsing.mokapot import MokapotPSMParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum
from ms2ml.type_defs import PathLike


@dataclass
class _MZML_AdapterBuffer:
    """Buffer of mzml adapters.

    It implements a form of buffer where adapters are stored.
    If the buffer capacity is reached, the adapter accessed "the longest time ago"
    is removed from the buffer.

    This is used only because keeping the indices of the mzml files in memory
    can get to be expensive and since the percolator file is sorted by file name
    we could in theory preserve the latest mzml adapters in memory for performance.
    """

    config: Config
    adapters: dict[str, MZMLAdapter] = None
    adapter_latest_accesses: dict[str, int] = None
    adapter_access_count: dict[str, int] = None
    accesses: int = 0
    max_size: int = 10
    access_keep: int = 20

    def __post_init__(self):
        self.adapters = {}
        self.adapter_access_count = {}
        self.adapter_latest_accesses = {}

    def __getitem__(self, key: str) -> MZMLAdapter:
        if key not in self.adapters:
            logger.info(f"Adding mzml adapter to buffer key={key}")
            self.adapters[key] = MZMLAdapter(key, self.config)
            self.adapter_latest_accesses[key] = 0
            self.adapter_access_count[key] = 0
        self.accesses += 1
        self.adapter_access_count[key] += 1
        self.adapter_latest_accesses[key] = self.accesses
        if len(self.adapters) > self.max_size:
            # remove the adapter that was accessed the longest time ago
            to_remove = [
                k
                for k, v in self.adapter_latest_accesses.items()
                if v < self.accesses - self.access_keep
            ]
            for key in to_remove:
                logger.info(f"Removing mzml adapter from buffer key={key}")
                del self.adapters[key]
                del self.adapter_latest_accesses[key]
        return self.adapters[key]

    def __contains__(self, key: str) -> bool:
        return key in self.adapters


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

        self.mzml_adapters = _MZML_AdapterBuffer(config=config)
        self.rawfile_mappings = {}

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

        if spec_dict["rawfile"] not in self.rawfile_mappings:
            # self.mzml_adapters
            self.rawfile_mappings[spec_dict["rawfile"]] = _find_raw_file(
                self.raw_file_locations, spec_dict["rawfile"]
            )

        spec = self.mzml_adapters[self.rawfile_mappings[spec_dict["rawfile"]]][
            spec_dict["spectrumindex"]
        ]

        if "precursorcharge" in spec_dict:
            spec_charge = spec_dict["precursorcharge"]
        elif (spec_charge := spec.precursor_charge) is not None:
            pass
        else:
            spec_charge = "2"
            logger.error(
                "No precursor charge found in the mokapot file or the raw data.",
                "Will default to 2 for now.",
                "Consider adding a 'charge_state' column to your mokapot file.",
            )

        seq = f"{spec_dict['peptidesequence']}/{spec_charge}"
        pep = Peptide.from_proforma_seq(seq, config=self.config)
        spec = spec.annotate(pep)
        annot_intensity = sum(spec.fragment_intensities.values())
        tot_intensity = spec.tic
        annot_frac = annot_intensity / tot_intensity

        Q_THRESHOLD = 0.01
        if annot_frac < Q_THRESHOLD:
            logger.warning(
                f"Only {annot_frac:.2%} of the intensity is annotated for {spec}."
            )
        spec.extras.update(spec_dict)

        return spec
