from typing import Callable, Iterator, Optional

from ms2ml.config import Config
from ms2ml.data.parsing.msp import MSPParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum
from ms2ml.utils.tensor_utils import pad_collate

from .base import BaseAdapter


class MSPAdapter(BaseAdapter):
    def __init__(
        self,
        config: Config,
        file: Optional[str] = None,
        in_hook: Optional[Callable] = None,
        out_hook: Optional[Callable] = None,
        collate_fn: Callable = pad_collate,
    ):
        BaseAdapter.__init__(
            self,
            config=config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        self.parser = MSPParser(file)

    def _to_elem(self, spec_dict) -> AnnotatedPeptideSpectrum:
        """
        Converts a dictionary to an AnnotatedPeptideSpectrum object.
        It is implicitly called by the _process_elem method.

        Args:
            spec_dict (dict): A dictionary containing the information for a spectrum.
                usually the result of parsing an msp file using their MSP parser.
        """
        pep = Peptide.from_proforma_seq(spec_dict["header"]["Name"], config=self.config)
        header = spec_dict["header"]
        spec = AnnotatedPeptideSpectrum(
            mz=spec_dict["peaks"]["mz"],
            intensity=spec_dict["peaks"]["intensity"],
            ms_level=2,  # Is this a valid assumption?
            precursor_peptide=pep,
            precursor_mz=0 if "PrecursorMz" not in header else header["PrecursorMz"],
            precursor_charge=pep.charge,
            extras=None if "Comment" not in header else header["Comment"],
            config=self.config,
        )

        return spec

    def parse_text(self, text) -> Iterator[AnnotatedPeptideSpectrum]:
        for spec in self.parser.parse_text(text):
            yield self._process_elem(spec)

    def parse(self) -> Iterator[AnnotatedPeptideSpectrum]:
        for spec in self.parser.parse():
            out = self._process_elem(spec)
            yield out

    def parse_file(self, file) -> Iterator[AnnotatedPeptideSpectrum]:
        for spec in self.parser.parse_file(file):
            yield self._process_elem(spec)

    def batch(self, batch_size: int) -> Iterator[AnnotatedPeptideSpectrum]:
        yield from super().batch(self.parse(), batch_size=batch_size)
