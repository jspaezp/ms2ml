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


def test_fasta_adapter_with_deltamass_mods(shared_datadir):
    config = Config(
        mod_mode="delta_mass",
        mod_fixed_mods=("[+22.2222]@C",),
        mod_variable_mods={"[+42.2222]": ["C"]},
        encoding_mod_alias={},
        precursor_charges=(2, 3),
    )
    adapter = FastaAdapter(
        shared_datadir / "fasta/P09038.fasta", config=config, allow_modifications=True
    )
    out = list(adapter.parse())
    out_mod_len = len(out)
    proformas = [x.to_proforma() for x in out]

    seen = set()
    dupes = []

    for x in proformas:
        if x in seen:
            dupes.append(x)
        else:
            seen.add(x)

    dupes.sort()
    unique_proforma_len = len(seen)
    explicit_fixed_mods = [x for x in seen if "C[+22" in x]

    assert len(explicit_fixed_mods) == 0, f"Fixed mods found, {explicit_fixed_mods}"
    assert out_mod_len == unique_proforma_len, f"Duplicate proformas found, {dupes}"
    assert "ALPGGRLGGR/2" in seen
    assert "ALPGGRLGGR/3" in seen
    assert "<[+22.2222]@C>RLYC[+42.2222]KNGGFFLR/2" in seen
    assert "<[+22.2222]@C>RLYCKNGGFFLR/2" in seen
