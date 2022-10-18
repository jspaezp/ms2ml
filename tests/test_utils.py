import numpy as np

from ms2ml.utils import (
    allign_intensities,
    annotate_peaks,
    find_matching,
    find_matching_sorted,
)


def test_matching_sorted_works():
    A = [0, 2, 4, 5]
    B = [0, 0, 0, 5]

    outs1 = [0, 0, 0, 2, 3]
    outs2 = [0, 1, 2, 3, 3]

    ia, ib = find_matching_sorted(A, B, 1)
    assert np.allclose(outs1, ia)
    assert np.allclose(outs2, ib)

    A = [0, 2, 4, 5, 5.1, 5.11, 5.111]
    B = [0.1, 0.11, 0.111, 5]

    outs1 = [0, 0, 0, 2, 3, 4, 5, 6]
    outs2 = [0, 1, 2, 3, 3, 3, 3, 3]
    ia, ib = find_matching_sorted(A, B, 1)
    assert np.allclose(outs1, ia)
    assert np.allclose(outs2, ib)


def test_matching_scrambled_works():
    A = [0, 2, 4, 5, 5.1, 5.11, 5.111]
    B = [0.1, 0.11, 0.111, 5]

    ia, ib = find_matching(A, B[::-1], 1)
    outs1 = [0, 0, 0, 2, 3, 4, 5, 6]
    outs2 = [3, 2, 1, 0, 0, 0, 0, 0]
    assert np.allclose(outs1, ia)
    assert np.allclose(outs2, ib)


"""
# From proteometools FTMS_HCD_30_annotated_2019-11-12.msp
Name: AAESLQRAEATNAELER/2
MW: 1857.9177
Comment: Parent=929.9661 Mods=0 Modstring=AAESLQRAEATNAELER///2 iRT=27.26 NCE=30.0 note="I removed all y ions and adducts and multicharge"
Num peaks: 32
143.0814	24869	"b2/3.94ppm"
272.123	20420	"b3/3.84ppm"
359.1557	15797	"b4/0.92ppm"
472.2363	14599	"b5/7.9ppm"
756.3931	10386	"b7/9.09ppm"
956.486	35345	"b9/-6.3ppm"
1027.5098	15860	"b10/7.04ppm"
1242.6079	40381	"b12/-0.73ppm"
1313.634	17590	"b13/7.59ppm"
1442.684	58765	"b14/2.11ppm"
1555.7889	12996	"b15/-11.53ppm"
1684.7969	41838	"b16/10.16ppm"

Name: AAESLQRAEATNAELER/3
MW: 1857.9177
Comment: Parent=619.9774 Mods=0 Modstring=AAESLQRAEATNAELER///3 iRT=27.34 NCE=30.0 note="I removed all y ions and adducts and multicharge"
Num peaks: 54
143.0816	1533755	"b2/3.09ppm"
272.1245	277764	"b3/-1.66ppm"
359.156	123107	"b4/-0.02ppm"
600.2936	67887	"b6/9.03ppm"
756.4019	225630	"b7/-2.53ppm"
827.4403	158163	"b8/-4ppm"
956.4791	382662	"b9/0.98ppm"
1027.5178	218633	"b10/-0.8ppm"
1128.5613	68407	"b11/2.41ppm"
1242.6085	115289	"b12/-1.22ppm"

# From FTMS_HCD_20_annotated_2019-11-12.msp
Name: AAESLQRAEATNAELER/2
MW: 1857.9177
Comment: Parent=929.9661 Mods=0 Modstring=AAESLQRAEATNAELER///2 iRT=27.38 NCE=20.0 note="I removed all y ions and adducts"
Num peaks: 8
956.4833	25009	"b9/-3.49ppm"
1442.6864	17228	"b14/0.41ppm"
1684.8269	22282	"b16/-7.66ppm"
"""  # noqa: E501


def test_matching_align_intensity_works():
    sample_spectra = [
        """143.0814	24869	"b2/3.94ppm"
        272.123	20420	"b3/3.84ppm"
        359.1557	15797	"b4/0.92ppm"
        472.2363	14599	"b5/7.9ppm"
        756.3931	10386	"b7/9.09ppm"
        956.486	35345	"b9/-6.3ppm"
        1027.5098	15860	"b10/7.04ppm"
        1242.6079	40381	"b12/-0.73ppm"
        1313.634	17590	"b13/7.59ppm"
        1442.684	58765	"b14/2.11ppm"
        1555.7889	12996	"b15/-11.53ppm"
        1684.7969	41838	"b16/10.16ppm" """,
        """ 143.0816	1533755	"b2/3.09ppm"
        272.1245	277764	"b3/-1.66ppm"
        359.156	123107	"b4/-0.02ppm"
        600.2936	67887	"b6/9.03ppm"
        756.4019	225630	"b7/-2.53ppm"
        827.4403	158163	"b8/-4ppm"
        956.4791	382662	"b9/0.98ppm"
        1027.5178	218633	"b10/-0.8ppm"
        1128.5613	68407	"b11/2.41ppm"
        1242.6085	115289	"b12/-1.22ppm" """,
        """ 956.4833	25009	"b9/-3.49ppm"
        1442.6864	17228	"b14/0.41ppm"
        1684.8269	22282	"b16/-7.66ppm" """,
    ]

    sample_spectra = [
        [y.strip().split("\t") for y in x.split("\n")] for x in sample_spectra
    ]
    mzs = [np.array([y[0] for y in x], dtype=float) for x in sample_spectra]
    ints = [np.array([y[1] for y in x], dtype=float) for x in sample_spectra]
    [[y[2] for y in x] for x in sample_spectra]

    outs = annotate_peaks(mzs[0], mzs[2])
    assert outs[0].shape == outs[1].shape

    outs = allign_intensities(mzs[0], mzs[2], ints[0], ints[2])
    assert outs[0].shape == outs[1].shape
