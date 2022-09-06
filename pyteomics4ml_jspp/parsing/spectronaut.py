from io import StringIO
from os import PathLike
from typing import Iterator

import pandas as pd

from .base import BaseParser


class SpectronautLibrary(BaseParser):
    def __init__(self) -> None:
        super().__init__()

    def parse_file(self, file: PathLike) -> Iterator:
        self.df = pd.read_csv(file)
        for i, x in self.df.groupby(
            ["ModifiedPeptide", "PrecursorCharge", "PrecursorMz"]
        ):

            out = x.to_dict("tight")
            out.update(
                {"ModifiedPeptide": i[0], "PrecursorCharge": i[1], "PrecursorMz": i[2]}
            )
            yield out

    def parse_text(self, text: str) -> Iterator:
        for x in self.parse_file(StringIO(text)):
            yield x
