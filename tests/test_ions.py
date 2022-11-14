import pytest

from ms2ml import Config, Peptide


def test_getting_ions():
    config = Config(ion_charges=(1, 2, 3))
    pep = Peptide.from_sequence("MPEP/3", config=config)
    out = pep.ion_dict

    with pytest.raises(KeyError) as _:
        # This makes sure that there is no y4 ion reported for a
        # peptide of length 4
        out["z1y4"]

    rounded_out = [int(x.mass) for x in out.values()]
    expected_mpep = [
        132,
        229,
        358,
        116,
        245,
        342,
        66,  # z 2+ ions
        115,
        179,
        58,
        123,
        171,
        44,  # z 3+ ions
        77,
        120,
        39,
        82,
        114,
    ]

    for x in expected_mpep:
        assert x in rounded_out


# Ground truth as defined in
# http://db.systemsbiology.net:8080/proteomicsToolkit/FragIonServlet.html
parametrized_peptides = (
    (
        "PEPICNK/2",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "PEPIC[Carbamidomethyl]NK/2",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "PEPIC[Carbamidomethyl]NK/2",
        {"z1b6": 711.31307, "z1b4": 437.23951, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
    (
        "[Acetyl]-PEPIC[Carbamidomethyl]NK/2",
        {"z1b6": 753.32364, "z1b4": 479.25007, "z1y4": 534.27048, "z1y2": 261.15578},
    ),
)


@pytest.mark.parametrize("sequence,ion_dict", parametrized_peptides)
def test_proforma_ion_parsing(sequence, ion_dict):
    config = Config(
        ion_charges=(1,),
        ion_naming_convention="z{ion_charges}{ion_series}{fragment_positions}",
    )
    pep = Peptide.from_sequence(sequence, config=config)
    out = pep.ion_dict

    for ion_type, mass in ion_dict.items():
        assert ion_type in out
        assert out[ion_type].mass == pytest.approx(mass, rel=1e-4)
