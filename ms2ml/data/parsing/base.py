from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Iterator, Optional, TextIO, Union


class BaseParser(ABC):
    """Base class for parsers."""

    file: Optional[Union[PathLike[Any], TextIO]]

    @abstractmethod
    def parse_file(self, file: Union[PathLike[Any], TextIO]) -> Iterator:
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

        for item in self.parse_file(self.file):
            yield item
