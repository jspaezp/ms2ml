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


variable_pos_mods_inputs = [
    {
        "input_sequence": "AMAMK",
        "mode": "unimod",
        "static_mods": (),
        "dynamic_mods": {},
        "expected_out": [
            "AMAMK",
        ],
    },
    {
        "input_sequence": "ACORNS",
        "mode": "unimod",
        "static_mods": ("[UNIMOD:4]@C",),
        "dynamic_mods": {
            "[UNIMOD:21]": [
                "S",
                "T",
                "Y",
            ],
        },
        "expected_out": [
            "<[UNIMOD:4]@C>ACORNS",
            "<[UNIMOD:4]@C>ACORNS[UNIMOD:21]",
        ],
    },
    {
        "input_sequence": "AMAMK",
        "mode": "delta_mass",
        "static_mods": (),
        "dynamic_mods": {
            "[+42.2222]": [
                "M",
            ],
        },
        "expected_out": [
            "AMAMK",
            "AMAM[+42.2222]K",
            "AM[+42.2222]AMK",
            "AM[+42.2222]AM[+42.2222]K",
        ],
    },
    {
        "input_sequence": "ACORNS",
        "mode": "delta_mass",
        "static_mods": ("[+22.2222]@C",),
        "dynamic_mods": {
            "[+42.2222]": [
                "C",
            ],
        },
        "expected_out": [
            "<[+22.2222]@C>ACORNS",
            "<[+22.2222]@C>AC[+42.2222]ORNS",
        ],
    },
    {
        "input_sequence": "ACORNS",
        "mode": "delta_mass",
        "static_mods": ("[+22.2222]@C",),
        "dynamic_mods": {},
        "expected_out": [
            "<[+22.2222]@C>ACORNS",
        ],
    },
]


@pytest.mark.parametrize("input", variable_pos_mods_inputs)
def test_variable_possible_mods(input):
    config = Config(
        mod_mode=input["mode"],
        mod_fixed_mods=input["static_mods"],
        mod_variable_mods=input["dynamic_mods"],
        encoding_mod_alias={},
    )
    pep = Peptide.from_sequence(input["input_sequence"], config=config)
    out = pep.get_variable_possible_mods()
    out = [x.to_proforma() for x in out]
    out.sort()

    unexpected = set(out) - set(input["expected_out"])
    assert len(unexpected) == 0, f"Unexpected -> {unexpected}"

    missing = set(input["expected_out"]) - set(out)
    assert len(missing) == 0, f"Missing -> {missing}"
