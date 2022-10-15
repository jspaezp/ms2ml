from __future__ import annotations

import re
from io import StringIO
from os import PathLike
from typing import Any, Iterator, TextIO

from .base import BaseParser


class PinParser(BaseParser):

    NUMERIC_REGEX = re.compile(r"^-?(([0-9]*)|(([0-9]*)\.([0-9]*)))$")

    # sample SpecId -> sample_tiny_hela_10039_2_5
    # It is {raw_file without extension}_{scan_number}_{charge}_{match_rank}
    SPECID_REGEX = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)?")

    # Sample Peptide -> K.AAASGK.A
    # it is {prev.aa}.{peptide}.{next.aa}
    PEPTIDE_REGEX = re.compile(r"^(.)+\.(.+)\.(.)+$")

    def __init__(self, file=None) -> None:
        BaseParser.__init__(self)
        self.file = file

    def parse_file(self, file: TextIO | PathLike[Any]) -> Iterator:
        """

        These are the columns in a comet pin file:
            SpecId
            Label
            ScanNr
            ExpMass
            CalcMass
            lnrSp
            deltLCn
            deltCn
            lnExpect
            Xcorr
            Sp
            IonFrac
            Mass
            PepLen
            Charge1
            Charge2	...
            Charge6
            enzN
            enzC
            enzInt
            lnNumSP
            dM
            absdM
            Peptide
            Proteins

        """
        with open(file) as f:
            header = next(f).strip().split("\t")
            for line in f:
                line = line.strip().split("\t")
                line2 = line[: (len(header) - 1)]
                line2 = [self._maybe_numeric(x) for x in line2]
                line2.append(line[len(header) - 1 :])
                out = dict(zip(header, line2))

                spec_id = out["SpecId"]
                match = self.SPECID_REGEX.match(spec_id)
                raw_file, index, charge, rank = match.groups(spec_id)
                out["RawFile"] = raw_file
                out["SpectrumIndex"] = int(index)
                out["PrecursorCharge"] = int(charge)
                out["MatchRank"] = int(rank)

                peptide = out["Peptide"]
                match = self.PEPTIDE_REGEX.match(peptide)
                prev_aa, peptide, next_aa = match.groups(peptide)
                out["PeptideSequence"] = peptide
                out["PreviousAminoAcid"] = prev_aa
                out["NextAminoAcid"] = next_aa

                yield out

    def parse_text(self, text: str) -> Iterator:
        yield from self.parse_file(StringIO(text))

    def parse(self) -> Iterator:
        if self.file is None:
            raise ValueError("No file specified")

        yield from self.parse_file(self.file)

    def _maybe_numeric(self, in_str) -> str | float:
        if self.NUMERIC_REGEX.match(in_str):
            if "." in in_str:
                return float(in_str)
            else:
                return int(in_str)
        return in_str


if __name__ == "__main__":
    from pprint import pprint

    foo = PinParser("tests/data/pin/sample_tiny_hela.pin")
    foo2 = next(foo.parse())
    pprint(foo2)
