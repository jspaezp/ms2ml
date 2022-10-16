from ms2ml.config import Config
from ms2ml.data import adapters, parsing
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum, Spectrum

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version("ms2ml")


__all__ = [
    "adapters",
    "parsing",
    "Config",
    "Spectrum",
    "Peptide",
    "AnnotatedPeptideSpectrum",
]
