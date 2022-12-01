from pathlib import Path
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
        df.columns = map(str.lower, df.columns)
        logger.debug(f"Loaded {len(df)} rows from {file}")
        df = df[df["mokapot q-value"] <= self.max_q]
        logger.debug(f"Filtered to {len(df)} rows with q-value <= {self.max_q}")

        for row_dict in df.to_dict(orient="records"):
            if "mzml_file" in row_dict and "scannr" in row_dict:
                row_dict["rawfile"] = Path(row_dict["mzml_file"]).stem
                row_dict["spectrumindex"] = int(row_dict["scannr"])
                row_dict["precursorcharge"] = int(row_dict["charge_state"])
                row_dict["matchrank"] = 1
            elif "specid" in row_dict:
                spec_id: str = str(row_dict["specid"])
                sid_match = self.SPECID_REGEX.match(spec_id)
                if sid_match is None:
                    raise ValueError(f"Could not parse specid {spec_id}")
                raw_file, index, charge, rank = sid_match.groups(spec_id)
                row_dict["rawfile"] = Path(raw_file).stem
                row_dict["spectrumindex"] = int(index)
                row_dict["precursorcharge"] = int(charge)
                row_dict["matchrank"] = int(rank)

            peptide = row_dict["peptide"]
            # TODO move parsing the peptide seq to a separate function
            # Bundle with pin parser
            if pep_match := self.PEPTIDE_REGEX.match(peptide):
                prev_aa, peptide, next_aa = pep_match.groups(peptide)
            else:
                prev_aa, next_aa = "", ""

            nterm_mod_match = self.NTERM_MOD_REGEX.match(peptide)
            if nterm_mod_match is not None:
                _, mod, peptide = nterm_mod_match.groups(peptide)
                peptide = f"{mod}-{peptide}"
            row_dict["peptidesequence"] = peptide
            row_dict["previousaminoacid"] = prev_aa
            row_dict["nextaminoacid"] = next_aa
            yield row_dict
