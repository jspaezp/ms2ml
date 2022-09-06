from typing import Callable

from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.parsing.msp import MSPParser
from pyteomics4ml_jspp.peptide import Peptide
from pyteomics4ml_jspp.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter


class MSPAdapter(MSPParser, BaseAdapter):
    def __init__(
        self, config: Config, in_hook: Callable = None, out_hook: Callable = None
    ):
        BaseAdapter.__init__(self, config=config, in_hook=in_hook, out_hook=out_hook)
        MSPParser.__init__(self)

    @staticmethod
    def _to_spec(spec_dict, config: Config) -> AnnotatedPeptideSpectrum:
        pep = Peptide.from_proforma_seq(spec_dict["header"]["Name"], config=config)
        spec = AnnotatedPeptideSpectrum(
            mz=spec_dict["peaks"]["mz"],
            intensity=spec_dict["peaks"]["intensity"],
            ms_level=2,  # Is this a valid assumption?
            precursor_peptide=pep,
            extras=spec_dict["header"]["Comment"],
        )

        return spec

    def parse_text(self, text):
        for spec in super().parse_text(text):
            yield self._process_spec(spec)
