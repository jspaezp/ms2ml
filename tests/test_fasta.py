from ms2ml.config import Config
from ms2ml.data.adapters.fasta import FastaAdapter
from ms2ml.data.parsing.fasta import FastaDataset


def test_fasta_parser(shared_datadir):
    fasta = FastaDataset(
        shared_datadir / "fasta/P09038.fasta",
        enzyme="trypsin",
        missed_cleavages=2,
        min_length=5,
        max_length=30,
    )
    out = list(fasta.parse())
    first_pep = out[0]
    assert first_pep["sequence"] == "YTSWYVALKRTGQYK"


def test_fasta_adapter(shared_datadir):
    # 6 is chosen so it ignores the first peptide
    config = Config(peptide_length_range=(6, 50), precursor_charges=(2, 3))
    adapter = FastaAdapter(shared_datadir / "fasta/P09038.fasta", config=config)
    out = list(adapter.parse())

    first_pep = out[0]
    assert first_pep.to_proforma() == "YTSWYVALKRTGQYK/2"


def test_fasta_adapter_with_mods(shared_datadir):
    # 6 is chosen so it ignores the first peptide
    config = Config(peptide_length_range=(6, 50), precursor_charges=(2, 3))
    adapter = FastaAdapter(
        shared_datadir / "fasta/P09038.fasta", config=config, allow_modifications=False
    )
    out_unmod_len = len(list(adapter.parse()))
    adapter = FastaAdapter(
        shared_datadir / "fasta/P09038.fasta", config=config, allow_modifications=True
    )
    out = list(adapter.parse())
    out_mod_len = len(out)

    first_pep = out[0]
    assert out_mod_len > out_unmod_len
    assert first_pep.stripped_sequence == "YTSWYVALKRTGQYK"
