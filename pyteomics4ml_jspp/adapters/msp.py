from typing import Callable

from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.parsing.msp import MSPParser
from pyteomics4ml_jspp.peptide import Peptide
from pyteomics4ml_jspp.spectrum import AnnotatedPeptideSpectrum


class MSPAdapter(MSPParser):
    def __init__(
        self, config: Config, in_hook: Callable = None, out_hook: Callable = None
    ):
        self.config = config
        self.in_hook = in_hook
        self.out_hook = out_hook
        super().__init__()

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

    def _process_spec(self, spec):
        spec = spec if self.in_hook is None else self.in_hook(spec)
        spec = self._to_spec(spec, config=self.config)
        spec = spec if self.out_hook is None else self.out_hook(spec)
        return spec

    def parse(self, text):
        parsed = super().parse(text)
        parsed = [self._process_spec(spec) for spec in parsed]
        return parsed
