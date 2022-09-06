from abc import ABC, abstractmethod
from os import PathLike
from typing import Iterator


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
