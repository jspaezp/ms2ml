from typing import Any, Callable, Iterator

from ms2ml.config import Config
from ms2ml.data.parsing.spectronaut import SpectronautLibraryParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter


class SpectronautAdapter(SpectronautLibraryParser, BaseAdapter):
    def __init__(
        self,
        config: Config,
        in_hook: Callable = None,
        out_hook: Callable = None,
        collate_fn: Callable[..., Any] = None,
    ):
        BaseAdapter.__init__(
            self,
            config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        SpectronautLibraryParser.__init__(self)

    def _to_elem(self, spec_dict) -> AnnotatedPeptideSpectrum:
        pep = Peptide.from_proforma_seq(spec_dict["LabeledPeptide"], config=self.config)

        spec = AnnotatedPeptideSpectrum(
            mz=spec_dict["FragmentMz"],
            intensity=spec_dict["RelativeIntensity"],
            ms_level=2,  # Is this a valid assumption?
            precursor_peptide=pep,
            precursor_mz=spec_dict["PrecursorMz"],
            precursor_charge=spec_dict["PrecursorCharge"],
            extras=spec_dict,
        )

        # The keys are ...
        # ['FragmentMz', 'RelativeIntensity', 'ModifiedPeptide',
        # 'LabeledPeptide', 'StrippedPeptide', 'PrecursorCharge',
        # 'PrecursorMz', 'iRT', 'FragmentNumber', 'FragmentType',
        # 'FragmentCharge', 'FragmentLossType']
        return spec

    def parse_file(self, file) -> Iterator[AnnotatedPeptideSpectrum]:
        for spec in super().parse_file(file):
            yield self._process_elem(spec)
