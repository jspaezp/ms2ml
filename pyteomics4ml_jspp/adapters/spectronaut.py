from typing import Callable

from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.parsing.spectronaut import SpectronautLibraryParser
from pyteomics4ml_jspp.peptide import Peptide
from pyteomics4ml_jspp.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter


class SpectronautAdapter(SpectronautLibraryParser, BaseAdapter):
    def __init__(
        self,
        config: Config,
        in_hook: Callable = None,
        out_hook: Callable = None,
        collate_fn: Callable = None,
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
            extras=spec_dict,
        )

        # The keys are ...
        # ['FragmentMz', 'RelativeIntensity', 'ModifiedPeptide',
        # 'LabeledPeptide', 'StrippedPeptide', 'PrecursorCharge',
        # 'PrecursorMz', 'iRT', 'FragmentNumber', 'FragmentType',
        # 'FragmentCharge', 'FragmentLossType']
        return spec

    def parse_file(self, file) -> AnnotatedPeptideSpectrum:
        for spec in super().parse_file(file):
            yield self._process_elem(spec)
