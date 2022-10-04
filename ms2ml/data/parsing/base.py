from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Iterator, TextIO


class BaseParser(ABC):
    """Base class for parsers."""

    file: PathLike[Any] | TextIO | None

    @abstractmethod
    def parse_file(self, file: PathLike[Any] | TextIO) -> Iterator:
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
        if self.file is None:
            raise ValueError("No file specified")

        yield from self.parse_file(self.file)
