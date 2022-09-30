"""Provides the data adapters for the ms2ml package.

The data adapters are used to load the data from the different data sources
into a common format. The data adapters use the parsers to load the
data into the common series of objects.
"""

from .base import BaseAdapter
from .msp import MSPAdapter
from .spectronaut import SpectronautAdapter  # noqa F401

__all__ = ["BaseAdapter", "MSPAdapter", "SPectronautAdapter"]
