from io import StringIO
from os import PathLike
from typing import Any, Iterator, TextIO, Union

import pandas as pd

from .base import BaseParser


class SpectronautLibraryParser(BaseParser):
    def __init__(self) -> None:
        BaseParser.__init__(self)

    def parse_file(self, file: Union[TextIO, PathLike[Any]]) -> Iterator:
        self.df = pd.read_csv(file)
        for i, x in self.df.groupby(
            ["ModifiedPeptide", "PrecursorCharge", "PrecursorMz"]
        ):
            out = x.to_dict("list")
            out.update(
                {"ModifiedPeptide": i[0], "PrecursorCharge": i[1], "PrecursorMz": i[2]}
            )
            # The keys are ...
            # ['FragmentMz', 'RelativeIntensity', 'ModifiedPeptide',
            # 'LabeledPeptide', 'StrippedPeptide', 'PrecursorCharge',
            # 'PrecursorMz', 'iRT', 'FragmentNumber', 'FragmentType',
            # 'FragmentCharge', 'FragmentLossType']
            yield out

    def parse_text(self, text: str) -> Iterator:
        for x in self.parse_file(StringIO(text)):
            yield x