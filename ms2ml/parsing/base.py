from abc import ABC, abstractmethod
from os import PathLike
from typing import Iterator


class BaseParser(ABC):
    """Base class for parsers."""

    @classmethod
    @abstractmethod
    def parse_file(self, file: PathLike) -> Iterator:
        """Parse a file.

        Parameters
        ----------
        file : PathLike
            Path to the file to parse

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """

    def __iter__(self) -> Iterator:
        """Parse the database.

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        for item in self.parse_file(self.file):
            yield item
