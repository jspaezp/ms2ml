import pytest

from ms2ml.isoform_utils import _get_mod_isoforms, get_mod_isoforms, get_mod_possible


def test_mod_isoform_works():
    seq_elems = [
        ("n_term", None),
        ("S", None),
        ("M", None),
        ("S", ["[U:21]"]),
        ("C", ["[U:4]"]),
        ("c_term", None),
    ]
    mod = "[U:21]"
    aas = "STY"
    out = _get_mod_isoforms(seq_elems, mod, aas)
    assert len(out) == 2

    seq_elems = [
        ("n_term", None),
        ("S", None),
        ("M", None),
        ("M", "[U:35]"),
        ("S", "[U:21]"),
        ("C", "[U:4]"),
        ("c_term", None),
    ]
    mods = {"[U:21]": "STY", "[U:35]": "M"}
    out = get_mod_isoforms(seq_elems, mods)
    assert len(out) == 4


mod_possible_params = [
    [
        [
            ("n_term", None),
            ("M", None),
            ("M", None),
            ("C", "[U:4]"),
            ("c_term", None),
        ],
        {"[U:21]": "STY", "[U:35]": "M"},
        4,
    ],
    [
        [
            ("n_term", None),
            ("M", None),
            ("M", "[U:35]"),
            ("C", "[U:4]"),
            ("c_term", None),
        ],
        {"[U:21]": "STY", "[U:35]": "M"},
        4,
    ],
    [
        [
            ("n_term", None),
            ("M", None),
            ("M", "[U:35]"),
            ("C", "[U:4]"),
            ("c_term", None),
        ],
        {"[U:21]": "STY"},
        1,
    ],
    [
        [
            ("n_term", None),
            ("M", None),
            ("S", None),
            ("M", "[U:35]"),
            ("C", "[U:4]"),
            ("c_term", None),
        ],
        {"[U:21]": "STY"},
        2,
    ],
    [
        [
            ("n_term", None),
            ("S", None),
            ("M", None),
            ("M", "[U:35]"),
            ("C", "[U:4]"),
            ("c_term", None),
        ],
        {"[U:21]": "STY", "[U:35]": "M"},
        8,
    ],
]


@pytest.mark.parametrize("seq_elems, mods, expected_num", mod_possible_params)
def test_mod_possible_works(seq_elems, mods, expected_num):
    out = get_mod_possible(seq_elems, mods)
    assert len(out) == expected_num

    for y in out:
        assert [x[0] for x in y] == [x[0] for x in seq_elems]
