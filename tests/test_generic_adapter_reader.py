import pytest

from ms2ml import AnnotatedPeptideSpectrum, Peptide, Spectrum
from ms2ml.config import Config
from ms2ml.data.adapters import read_data

annotated_spectrum_adapter_files = [
    # "pepxml/sample_tiny_hela.pep.xml",
    [
        "msp/head_FTMS_HCD_20_annotated_2019-11-12_filtered.msp",
        AnnotatedPeptideSpectrum,
    ],
    ["mokapot/mokapot.psms.txt", AnnotatedPeptideSpectrum],
    ["mokapot/mokapot.peptides.txt", AnnotatedPeptideSpectrum],
    ["bibliospec/Firstexp.blib", AnnotatedPeptideSpectrum],
    ["encyclopedia/pan_human_library_600to603.dlib", AnnotatedPeptideSpectrum],
    ["pin/sample_tiny_hela.pin", AnnotatedPeptideSpectrum],
    ["mzml/sample_tiny_hela.mzML", Spectrum],
    ["fasta/P09038.fasta", Peptide],
]


@pytest.mark.parametrize("file,expectedclass", annotated_spectrum_adapter_files)
def test_annot_spectrum_adapter(file, expectedclass, shared_datadir):
    """Test the annotated spectrum adapter."""
    config = Config()
    adapter = read_data(shared_datadir / file, config=config)

    out = next(iter(adapter.parse()))
    assert isinstance(out, expectedclass)

    # This makes sure it terminates graciously
    for out in adapter.parse():
        assert isinstance(out, expectedclass)
