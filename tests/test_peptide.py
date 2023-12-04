import pytest

from ms2ml import Config, Peptide


def test_peptide_raises_error_when_no_charge():
    config = Config()
    pep = Peptide.from_sequence("MPEP", config=config)

    with pytest.raises(ValueError) as error:
        pep.ion_dict

    assert "Peptide charge is not set" in str(error.value)

    config = Config()
    pep = Peptide.from_sequence("MPEP/2", config=config)
    pep.ion_dict


@pytest.mark.parametrize("delta_mass", [False, True])
def test_peptide_proforma(delta_mass: bool):
    if delta_mass:
        config = Config(
            mod_mode="delta_mass",
            mod_fixed_mods=("[+22.2222]@C",),
            mod_variable_mods={"[+22.2222]": ["C"]},
            encoding_mod_alias={},
        )
        sequences = [
            {
                "input_sequence": "LESLIEK",
                "expected_out": "LESLIEK",
                "expected_mass": 830.4749342093,
            },
            {
                "input_sequence": "ACORNS",
                "expected_out": "<[+22.2222]@C>ACORNS",
                "expected_mass": 786.3806 + 22.2222,
            },
            {
                "input_sequence": "<[+22.2222]@C>ACORNS",
                "expected_out": "<[+22.2222]@C>ACORNS",
                "expected_mass": 786.3806 + 22.2222,
            },
        ]
    else:
        config = Config(
            mod_mode="unimod",
            mod_fixed_mods=("[UNIMOD:4]@C",),
            mod_variable_mods={
                "[UNIMOD:35]": [
                    "M",
                ],
                "[UNIMOD:21]": [
                    "S",
                    "T",
                    "Y",
                ],
            },
            encoding_mod_alias={},
        )
        sequences = [
            {
                "input_sequence": "LESLIEK",
                "expected_out": "LESLIEK",
                "expected_mass": 830.4749342093,
            },
            {
                "input_sequence": "ACORNS",
                "expected_out": "<[UNIMOD:4]@C>ACORNS",
                "expected_mass": 843.4020,
            },
            {
                "input_sequence": "<[UNIMOD:4]@C>ACORNS",
                "expected_out": "<[UNIMOD:4]@C>ACORNS",
                "expected_mass": 843.4020,
            },
        ]

    for seq in sequences:
        pep = Peptide.from_sequence(seq["input_sequence"], config=config)
        print(pep.to_proforma())
        assert pep.to_proforma() == seq["expected_out"]
        assert int(1_000 * pep.mass) == int(1_000 * seq["expected_mass"])
