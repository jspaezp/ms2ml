import pytest

from ms2ml import Config


@pytest.mark.parametrize("delta_mass", [False, True])
def test_fixed_mod_validation(delta_mass):
    if delta_mass:
        valid_config_args = [
            {
                "mod_mode": "delta_mass",
                "mod_fixed_mods": ("[+22.2222]@C",),
                "mod_variable_mods": {"[+22.2222]": ["C"]},
                "encoding_mod_alias": {},
            },
            {
                "mod_mode": "delta_mass",
                "mod_fixed_mods": ("[+22.2222]@C",),
                "mod_variable_mods": {"[+22.2222]": ["C"]},
                "encoding_mod_alias": {},
            },
        ]
        non_valid_config_args = [
            {
                "mod_mode": "delta_mass",
                "mod_fixed_mods": "[+22.2222]@C",  # <- string instead of tuple
                "mod_variable_mods": {"[+22.2222]": ["C"]},
                "encoding_mod_alias": {},
            },
            {
                "mod_mode": "delta_mass",
                "mod_fixed_mods": ("[+22.2222]@C",),
                "mod_variable_mods": [{"[+22.2222]": ["C"]}],  # <- list instead of dict
                "encoding_mod_alias": {},
            },
        ]
    else:
        valid_config_args = [
            {
                "mod_mode": "unimod",
                "mod_fixed_mods": ("[UNIMOD:4]@C",),
                "mod_variable_mods": {
                    "[UNIMOD:35]": [
                        "M",
                    ],
                    "[UNIMOD:21]": [
                        "S",
                        "T",
                        "Y",
                    ],
                },
                "encoding_mod_alias": {},
            },
            {
                "mod_mode": "unimod",
                "mod_fixed_mods": ("[UNIMOD:4]@C",),
                "mod_variable_mods": {
                    "[UNIMOD:35]": [
                        "M",
                    ],
                    "[UNIMOD:21]": [
                        "S",
                        "T",
                        "Y",
                    ],
                },
                "encoding_mod_alias": {},
            },
        ]
        non_valid_config_args = [
            {
                "mod_mode": "unimod",
                "mod_fixed_mods": "[UNIMOD:4]@C",  # <- string instead of tuple
                "mod_variable_mods": {
                    "[UNIMOD:35]": [
                        "M",
                    ],
                    "[UNIMOD:21]": [
                        "S",
                        "T",
                        "Y",
                    ],
                },
                "encoding_mod_alias": {},
            },
            {
                "mod_mode": "unimod",
                "mod_fixed_mods": ("[UNIMOD:4]@C",),
                "mod_variable_mods": [
                    {
                        "[UNIMOD:35]": [
                            "M",
                        ],
                        "[UNIMOD:21]": [
                            "S",
                            "T",
                            "Y",
                        ],
                    }
                ],  # <- list instead of dict
                "encoding_mod_alias": {},
            },
        ]

    for config_args in valid_config_args:
        Config(**config_args)

    for config_args in non_valid_config_args:
        with pytest.raises(ValueError):
            Config(**config_args)
