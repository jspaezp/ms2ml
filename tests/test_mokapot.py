from ms2ml import AnnotatedPeptideSpectrum, Config
from ms2ml.data.adapters.mokapot import MokapotPSMAdapter
from ms2ml.data.parsing.mokapot import MokapotPSMParser


def test_mokapot_parser(shared_datadir):
    parser = MokapotPSMParser(shared_datadir / "mokapot" / "mokapot.psms.txt")
    elem = next(parser.parse())
    assert isinstance(elem, dict)
    assert elem["specid"] == "sample_tiny_hela_10044_3_1"
    assert elem["peptide"] == "R.HRLDLGEDYPSGK.K"
    assert elem["peptidesequence"] == "HRLDLGEDYPSGK"


def test_mokapot_adapter(shared_datadir):
    config = Config()
    raw_location = shared_datadir / "mzml"
    adapter = MokapotPSMAdapter(
        shared_datadir / "mokapot" / "mokapot.psms.txt",
        config=config,
        raw_file_locations=raw_location,
    )
    elem = next(adapter.parse())
    assert isinstance(elem, AnnotatedPeptideSpectrum)
