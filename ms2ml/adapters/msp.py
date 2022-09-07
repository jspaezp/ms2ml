from typing import Callable

from ms2ml.config import Config
from ms2ml.parsing.msp import MSPParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter


class MSPAdapter(MSPParser, BaseAdapter):
    def __init__(
        self,
        config: Config,
        in_hook: Callable = None,
        out_hook: Callable = None,
        collate_fn: Callable = None,
    ):
        BaseAdapter.__init__(
            self,
            config=config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        MSPParser.__init__(self)

    def _to_elem(self, spec_dict) -> AnnotatedPeptideSpectrum:
        pep = Peptide.from_proforma_seq(spec_dict["header"]["Name"], config=self.config)
        spec = AnnotatedPeptideSpectrum(
            mz=spec_dict["peaks"]["mz"],
            intensity=spec_dict["peaks"]["intensity"],
            ms_level=2,  # Is this a valid assumption?
            precursor_peptide=pep,
            extras=spec_dict["header"]["Comment"],
        )

        return spec

    def parse_text(self, text) -> AnnotatedPeptideSpectrum:
        for spec in super().parse_text(text):
            yield self._process_elem(spec)
