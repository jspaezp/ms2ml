from abc import ABC, abstractmethod
from typing import Iterator
from os import PathLike


class BaseParser(ABC):
    """
    Base class for parsers
    """

    @abstractmethod
    def parse_file(self, file: PathLike) -> Iterator:
        """
        Parse a file

        Parameters
        ----------
        file : PathLike
            Path to the file to parse

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        pass

    @abstractmethod
    def parse_text(self, text: str) -> Iterator:
        """
        Parse a chunk of text

        Parameters
        ----------
        text : str
            Chunk of text to parse

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        pass
