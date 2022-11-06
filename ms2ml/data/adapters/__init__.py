"""Provides the data adapters for the ms2ml package.

The data adapters are used to load the data from the different data sources
into a common format. The data adapters use the parsers to load the
data into the common series of objects.
"""

from .base import BaseAdapter
from .bibliospec import BiblioSpecAdapter
from .encyclopedia import EncyclopeDIAAdapter
from .msp import MSPAdapter
from .mzml import MZMLAdapter
from .spectronaut import SpectronautAdapter

__all__ = [
    "BaseAdapter",
    "MSPAdapter",
    "SpectronautAdapter",
    "MZMLAdapter",
    "BiblioSpecAdapter",
    "EncyclopeDIAAdapter",
]
