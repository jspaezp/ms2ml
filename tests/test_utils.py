import numpy as np

from ms2ml.utils import find_matching, find_matching_sorted


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
