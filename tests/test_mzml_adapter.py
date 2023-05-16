import pytest

from ms2ml.config import Config
from ms2ml.data.adapters.mzml import MZMLAdapter


def test_mzml_adapter_works(shared_datadir):
    file = shared_datadir / "mzml" / "sample_tiny_hela.mzML"

    spec_adapter = MZMLAdapter(
        file=str(file),
        config=Config(),
        collate_fn=list,
    )
    parsed = spec_adapter.parse()
    elem = next(parsed)

    assert elem.precursor_charge == 2
    assert elem.mz.shape == (46,)

    binned = elem.bin_spectrum(start=100, end=2000, binsize=0.5)
    assert binned.shape == (3799,)

    parsed = spec_adapter.parse()
    bundled = spec_adapter.bundle(parsed)
    assert len(bundled) == 24
    assert {x.ms_level for x in bundled} == {1, 2}


def test_bad_mzml(monkeypatch):
    """Test raise a no scan names error"""
    class DummyIndex:
        mapping = {"spectrum": [{"blah": 1}]}

    class DummyReader:
        index = DummyIndex

    monkeypatch.setattr(
        "ms2ml.data.adapters.mzml.read_mzml",
        lambda *args, **kwargs: DummyReader,
    )

    with pytest.raises(RuntimeError):
        MZMLAdapter("blah", Config(), collate_fn=list)
