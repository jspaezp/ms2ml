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
        self.only_targets = True
        self.max_q = 0.01

    def parse(self) -> Iterator:
        yield from self.parse_file(str(self.file))

    def parse_file(self, file) -> Iterator:
        df = pd.read_csv(str(file), sep="\t")
        df.columns = map(str.lower, df.columns)
        logger.debug(f"Loaded {len(df)} rows from {file}")
        df = df[df["mokapot q-value"] <= self.max_q]
        if self.only_targets:
            df = df[df["label"].astype(bool)]
        logger.warning(f"Filtered to {len(df)} rows with q-value <= {self.max_q}")

        if "mzml_file" in df.columns:
            df["rawfile"] = df["mzml_file"].apply(lambda x: Path(x).stem)
            df["spectrumindex"] = df["scannr"].astype(int)
            if "charge_state" in df.columns:
                df["precursorcharge"] = df["charge_state"].astype(int)
            elif "charge" in df.columns:
                df["precursorcharge"] = df["charge"].astype(int)
            else:
                logger.warning(
                    "No charge column found in mokapot file, "
                    "will try to infer from the raw data"
                )

            df["matchrank"] = 1

        elif "specid" in df.columns:
            extracts = df["specid"].str.extract(self.SPECID_REGEX)
            df["rawfile"] = extracts[0].apply(lambda x: Path(x).stem)
            df["spectrumindex"] = extracts[1].astype(int)
            df["precursorcharge"] = extracts[2].astype(int)
            df["matchrank"] = extracts[3].astype(int)

        df = df.sort_values(by=["rawfile", "spectrumindex", "matchrank"]).reset_index(
            drop=True
        )

        logger.info(f"Starting to parse the mokapot file (n={len(df)}).")
        for i, row_dict in enumerate(df.to_dict(orient="records")):
            # Log every 1k rows
            if i % 1000 == 0:
                logger.debug(f"Processed {i} rows.")

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
