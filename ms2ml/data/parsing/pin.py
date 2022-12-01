from __future__ import annotations

import re
import warnings
from io import StringIO
from typing import Iterator, TextIO

import pandas as pd

from ms2ml.type_defs import PathLike

from .base import BaseParser


class PinParser(BaseParser):
    NUMERIC_REGEX = re.compile(r"^-?(([0-9]*)|(([0-9]*)\.([0-9]*)))$")

    # sample SpecId -> sample_tiny_hela_10039_2_5
    # It is {raw_file without extension}_{scan_number}_{charge}_{match_rank}
    SPECID_REGEX = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)?")

    # Sample Peptide -> K.AAASGK.A, K.AAASGK.-, K.AAASGK.AQ
    # it is {prev.aa}.{peptide}.{next.aa}
    PEPTIDE_REGEX = re.compile(r"^([A-Z\-])+?\.(.+)\.([A-Z\-])+?$")

    # Sample -> n[229.1629]K[229.1629]VEIPGVATTASPSSEVGR/3
    NTERM_MOD_REGEX = re.compile(r"^(n)(\[[+-].*?\])(.*)$")

    def __init__(self, file=None) -> None:
        BaseParser.__init__(self)
        self.file = file
        self.parse_fun, self.pin_flavour = self._select_parser(file)

    def parse_file(self, file: TextIO | PathLike) -> Iterator:
        yield from self.parse_fun(file)

    def _select_parser(self, file: TextIO | PathLike) -> Iterator:
        comet_colnames = [
            "lnExpect",
            "Xcorr",
            "Sp",
            "IonFrac",
        ]

        sage_colnames = [
            "average_ppm",
            "calcmass",
            "charge",
            "delta_hyperscore",
        ]

        with open(file, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break

        if all([col in line for col in comet_colnames]):
            return self._comet_pin_parser, "comet"
        elif all([col in line for col in sage_colnames]):
            return self._sage_pin_parser, "sage"
        else:
            warnings.warn("Unknown specification of the pin format, defaulting to sage")
            return self._sage_pin_parser, "sage"

    def _sage_pin_parser(self, file: TextIO | PathLike) -> Iterator:
        """
        These are the cols for a sage pin file:

            {'Index': 0,
            'average_ppm': 2.1279066,
            'calcmass': 871.45197,
            'charge': 2,
            'delta_hyperscore': 3.4832007356480634,
            'delta_mass': 0.6310756,
            'delta_rt': 1.0,
            'discriminant_score': -0.25593093,
            'expmass': 871.4514,
            'hyperscore': 19.481596383491382,
            'isotope_error': 0.0,
            'label': 1,
            'longest_b': 0,
            'longest_y': 4,
            'longest_y_pct': 0.5714286,
            'matched_intensity_pct': 43.759075,
            'matched_peaks': 6,
            'missed_cleavages': 0,
            'num_proteins': 2,
            'peptide': 'IPDEELR',
            'peptide_len': 7,
            'poisson': -2.1477053087741056,
            'posterior_error': -11.716047,
            'predicted_rt': 0.0,
            'proteins': 'sp|Q99961|SH3G1_HUMAN;sp|Q99962|SH3G2_HUMAN',
            'q_value': 0.00295858,
            'rt': 10.488159, # is this 100% minutes?
            'scannr': 10676,
            'scored_candidates': 46,
            'specid': 0
        """
        df = pd.read_csv(file, sep="\t")
        for row in df.itertuples():
            yield row._asdict()

    def _comet_pin_parser(self, file: TextIO | PathLike) -> Iterator:
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
            Proteins # is tab separated ...
        """
        dfn = comet_pin_to_df(file)
        for row in dfn.itertuples():
            out = row._asdict()
            spec_id: str = out["SpecId"]
            sid_match = self.SPECID_REGEX.match(spec_id)
            if sid_match is None:
                raise ValueError(f"Could not parse SpecId {spec_id}")
            raw_file, index, charge, rank = sid_match.groups(spec_id)
            out["RawFile"] = raw_file
            out["SpectrumIndex"] = int(index)
            out["PrecursorCharge"] = int(charge)
            out["MatchRank"] = int(rank)

            peptide = out["Peptide"]
            pep_match = self.PEPTIDE_REGEX.match(peptide)
            if pep_match is None:
                raise ValueError(f"Could not parse peptide {peptide}")
            prev_aa, peptide, next_aa = pep_match.groups(peptide)
            nterm_mod_match = self.NTERM_MOD_REGEX.match(peptide)
            if nterm_mod_match is not None:
                _, mod, peptide = nterm_mod_match.groups(peptide)
                peptide = f"{mod}-{peptide}"
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


def comet_pin_to_df(file: TextIO | PathLike) -> pd.DataFrame:
    """
    Parses a comet pin file into a pandas dataframe
    """
    recs = []
    with open(file, encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            line = line.strip().split("\t")
            line2 = line[: (len(header) - 1)]
            line2.append(line[len(header) - 1 :])
            out = dict(zip(header, line2))
            recs.append(out)

    df = pd.DataFrame.from_records(recs)
    dfn = df.convert_dtypes()
    return dfn
