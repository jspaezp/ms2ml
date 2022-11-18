"""Provides the data adapters for the ms2ml package.

The data adapters are used to load the data from the different data sources
into a common format. The data adapters use the parsers to load the
data into the common series of objects.
"""

from pathlib import Path

from loguru import logger

from .base import BaseAdapter
from .bibliospec import BiblioSpecAdapter
from .encyclopedia import EncyclopeDIAAdapter
from .fasta import FastaAdapter
from .mokapot import MokapotPSMAdapter
from .msp import MSPAdapter
from .mzml import MZMLAdapter
from .pin import PinAdapter
from .spectronaut import SpectronautAdapter

EXTENSIONS = {
    ".msp": MSPAdapter,
    ".mzML": MZMLAdapter,
    ".mzml": MZMLAdapter,
    ".blib": BiblioSpecAdapter,
    ".dlib": EncyclopeDIAAdapter,
    ".elib": EncyclopeDIAAdapter,
    ".csv": SpectronautAdapter,
    ".fasta": FastaAdapter,
    ".fa": FastaAdapter,
    ".psms.txt": MokapotPSMAdapter,
    ".peptides.txt": MokapotPSMAdapter,
    ".pin": PinAdapter,
}


def read_data(path, config, *args, **kwargs):
    """Reads the data from the given path.

    Args:
        path (str): The path to the data.
        config (Config): The configuration object.

    Returns:
        list: A list of Peptide objects.
    """

    path = Path(path)
    for k, v in EXTENSIONS.items():
        if str(path).endswith(k):
            logger.info(f"Reading data from {path} using {v}")
            return v(file=path, config=config, *args, **kwargs)

    raise ValueError(f"Unknown file extension for file: {path}")


__all__ = [
    "BaseAdapter",
    "MSPAdapter",
    "SpectronautAdapter",
    "MZMLAdapter",
    "BiblioSpecAdapter",
    "EncyclopeDIAAdapter",
]
