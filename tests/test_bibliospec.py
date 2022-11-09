import numpy as np

from ms2ml import AnnotatedPeptideSpectrum, Config
from ms2ml.data.adapters.bibliospec import BiblioSpecAdapter
from ms2ml.data.parsing.bibliospec import BiblioSpecParser


def test_bibliospec_parser(shared_datadir):
    foo = BiblioSpecParser(shared_datadir / "bibliospec/Firstexp.blib")
    i = next(iter(foo))

    assert i["id"] == 1
    assert i["numPeaks"] == 14
    assert isinstance(i["peakIntensity"], np.ndarray)
    assert isinstance(i["peakMZ"], np.ndarray)


def test_bibliospec_adapter(shared_datadir):
    file = shared_datadir / "bibliospec/Firstexp.blib"

    adapter = BiblioSpecAdapter(file, config=Config())
    for i, spec in enumerate(adapter.parse()):
        if i == 0:
            assert len(spec.mz) == 14
        if i < 4:
            assert isinstance(spec, AnnotatedPeptideSpectrum)
            assert np.all(spec.intensity >= 0)
