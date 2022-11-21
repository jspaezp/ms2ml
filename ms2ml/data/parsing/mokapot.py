from typing import Iterator

import pandas as pd
from loguru import logger

from ms2ml.data.parsing import BaseParser
from ms2ml.data.parsing.pin import PinParser


class MokapotPSMParser(BaseParser):
    NUMERIC_REGEX = PinParser.NUMERIC_REGEX
    SPECID_REGEX = PinParser.SPECID_REGEX
    PEPTIDE_REGEX = PinParser.PEPTIDE_REGEX
    NTERM_MOD_REGEX = PinParser.NTERM_MOD_REGEX

    def __init__(self, file) -> None:
        BaseParser.__init__(self)
        self.file = file
        self.max_q = 0.01

    def parse(self) -> Iterator:
        yield from self.parse_file(str(self.file))

    def parse_file(self, file) -> Iterator:
        df = pd.read_csv(str(file), sep="\t")
        logger.debug(f"Loaded {len(df)} rows from {file}")
        df = df[df["mokapot q-value"] <= self.max_q]
        logger.debug(f"Filtered to {len(df)} rows with q-value <= {self.max_q}")

        for row_dict in df.to_dict(orient="records"):
            spec_id: str = str(row_dict["SpecId"])
            sid_match = self.SPECID_REGEX.match(spec_id)
            if sid_match is None:
                raise ValueError(f"Could not parse SpecId {spec_id}")
            raw_file, index, charge, rank = sid_match.groups(spec_id)
            row_dict["RawFile"] = raw_file
            row_dict["SpectrumIndex"] = int(index)
            row_dict["PrecursorCharge"] = int(charge)
            row_dict["MatchRank"] = int(rank)

            peptide = row_dict["Peptide"]
            # TODO move parsing the peptide seq to a separate function
            # Bundle with pin parser
            pep_match = self.PEPTIDE_REGEX.match(peptide)
            if pep_match is None:
                raise ValueError(f"Could not parse peptide {peptide}")
            prev_aa, peptide, next_aa = pep_match.groups(peptide)
            nterm_mod_match = self.NTERM_MOD_REGEX.match(peptide)
            if nterm_mod_match is not None:
                _, mod, peptide = nterm_mod_match.groups(peptide)
                peptide = f"{mod}-{peptide}"
            row_dict["PeptideSequence"] = peptide
            row_dict["PreviousAminoAcid"] = prev_aa
            row_dict["NextAminoAcid"] = next_aa
            yield row_dict
