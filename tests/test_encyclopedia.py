import numpy as np

from ms2ml.data.adapters import EncyclopeDIAAdapter
from ms2ml.data.parsing.encyclopedia import _extract_array, write_encyclopedia
from ms2ml.spectrum import AnnotatedPeptideSpectrum, Config


def test_encyclopedia_adapter(shared_datadir):
    file = shared_datadir / "encyclopedia/pan_human_library_600to603.dlib"

    adapter = EncyclopeDIAAdapter(file, config=Config())
    for i, spec in enumerate(adapter.parse()):
        assert isinstance(spec, AnnotatedPeptideSpectrum)
        assert np.all(spec.mz >= 0)
        assert np.all(spec.intensity >= 0)


def test_enciclopedia_decoding():
    mass_enc_len = 48
    mass_enc = b"x\x9c\x010\x00\xcf\xff@vq\xeat\x91\xe7\r@w\xd3iu3!\xaf@~\xe4\xc1\xc8\xacy\x12@\x84\x8a\xe2\x96\xba\xb9q@\x8a[\x85\x05\x90\x04<@\x8d\xf3\xbc3xI\xe0@>\x17Q"  # noqa
    mass_dec = _extract_array(mass_enc, "d")
    assert np.all(mass_dec > 0)
    assert len(mass_dec) * 8 == mass_enc_len

    int_enc_len = 24
    int_arr = b"x\x9csi\xdbs\xd6\xf5nP\x9a\xeb\xa7\xbf\xb3\xdcd\x1c\x18\\\xad\xb5\x8d]\x93\xf5\xd2\x00\x8d\xb7\t\xb8"  # noqa
    int_dec = _extract_array(int_arr, "f")
    assert np.all(int_dec >= 0)
    assert len(int_dec) * 4 == int_enc_len


def test_writting_enciclopedia(tmpdir):
    specs = [AnnotatedPeptideSpectrum._sample(), AnnotatedPeptideSpectrum._sample()]
    write_encyclopedia(tmpdir / "test.dlib", specs)
    adapter = EncyclopeDIAAdapter(tmpdir / "test.dlib", config=Config())
    for i, spec in enumerate(adapter.parse()):
        assert isinstance(spec, AnnotatedPeptideSpectrum)
        assert np.all(spec.mz >= 0)
        assert np.all(spec.intensity >= 0)

    assert i == 1