from ms2ml.config import Config
from ms2ml.data.adapters.pin import PinAdapter
from ms2ml.spectrum import AnnotatedPeptideSpectrum


def test_mzml_adapter_works(shared_datadir):
    file = shared_datadir / "pin" / "sample_tiny_hela.pin"
    raw_location = shared_datadir / "mzml"

    spec_adapter = PinAdapter(
        file=str(file),
        config=Config(),
        collate_fn=list,
        raw_file_locations=raw_location,
    )

    parsed = spec_adapter.parse()
    elem = next(parsed)

    assert isinstance(elem, AnnotatedPeptideSpectrum)
    assert elem.precursor_charge == 2
    assert elem.mz.shape == (46,)
    binned = elem.bin_spectrum(start=100, end=2000, binsize=0.5)
    assert binned.shape == (3799,)
    assert elem.encode_fragments().shape == (120,)

    col_names = [
        "ExpMass",
        "CalcMass",
        "lnrSp",
        "deltLCn",
        "deltCn",
        "lnExpect",
        "Xcorr",
        "Sp",
        "IonFrac",
        "Mass",
        "PepLen",
    ]

    for x in col_names:
        assert x in elem.extras, f"{x} not in extras"

    for _ in spec_adapter.parse():
        continue


"""
Disabled for this version while sage matures a little bit.
Right now it is changeing a lot the output to bundle it with
peptideshaker.
def test_mzml_adapter_works_sage(shared_datadir):
    file = shared_datadir / "pin" / "sample_tiny_hela.sage.pin"
    raw_location = shared_datadir / "mzml" / "sample_tiny_hela.mzML"

    spec_adapter = PinAdapter(
        file=str(file),
        config=Config(),
        collate_fn=list,
        raw_file_locations=raw_location,
    )

    parsed = spec_adapter.parse()
    elem = next(parsed)

    assert isinstance(elem, AnnotatedPeptideSpectrum)
    assert elem.precursor_charge == 2
    assert elem.mz.shape == (46,)
    binned = elem.bin_spectrum(start=100, end=2000, binsize=0.5)
    assert binned.shape == (3799,)
    assert elem.encode_fragments().shape == (120,)
"""
