import numpy as np

from ms2ml.parsing.bibliospec import BibliosPecParser


def test_bibliospec_parser(shared_datadir):
    foo = BibliosPecParser(shared_datadir / "bibliospec/Firstexp.blib")
    for i in foo:
        break

    assert i["id"] == 1
    assert i["numPeaks"] == 14
    assert isinstance(i["peakIntensity"], np.ndarray)
    assert isinstance(i["peakMZ"], np.ndarray)
