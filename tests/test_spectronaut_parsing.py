from pyteomics4ml_jspp.adapters.spectronaut import SpectronautAdapter
from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.data.utils import default_collate
from pyteomics4ml_jspp.parsing.spectronaut import SpectronautLibraryParser


def test_spectronaut_parsing_works(shared_datadir):
    file = shared_datadir / "prosit_outputs" / "peptidelist.generic"

    parser = SpectronautLibraryParser()
    spectra = parser.parse_file(file)

    elem = next(spectra)

    assert isinstance(elem, dict)


def test_spectronaut_adapter_works(shared_datadir):
    file = shared_datadir / "prosit_outputs" / "peptidelist.generic"

    def pre_hook(spec):
        spec["LabeledPeptide"] = spec["LabeledPeptide"][0].replace(
            "M(ox)", "M[Oxidation]"
        )
        return spec

    def post_hook(spec):
        return {
            "aa": spec.precursor_peptide.aa_to_onehot(),
            "mods": spec.precursor_peptide.mod_to_vector(),
        }

    spec_adapter = SpectronautAdapter(
        config=Config(),
        in_hook=pre_hook,
        out_hook=post_hook,
        collate_fn=default_collate,
    )
    parsed = spec_adapter.parse_file(file)
    elem = next(parsed)

    assert elem["aa"].shape == (11, 29)
    assert elem["mods"].shape == (11,)

    parsed = spec_adapter.parse_file(file)
    bundled = spec_adapter.bundle(parsed)

    assert bundled["aa"].shape == (3, 15, 29)
    assert bundled["mods"].shape == (3, 15)
