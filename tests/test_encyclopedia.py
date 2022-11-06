import numpy as np

from ms2ml import AnnotatedPeptideSpectrum, Config
from ms2ml.data.adapters import EncyclopeDIAAdapter


def test_encyclopedia_adapter(shared_datadir):
    file = shared_datadir / "encyclopedia/pan_human_library_600to603.dlib"

    adapter = EncyclopeDIAAdapter(file, config=Config())
    for i, spec in enumerate(adapter.parse()):
        assert isinstance(spec, AnnotatedPeptideSpectrum)
        assert np.all(spec.mz >= 0)
        assert np.all(spec.intensity >= 0)
