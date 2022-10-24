from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .constants import PROTON
from .types import MassError


def mz(mass: float | NDArray, charge: int) -> float | NDArray:
    return (mass + (charge * PROTON)) / charge


def ppm_diff(
    theoretical: NDArray | float, observed: NDArray | float
) -> NDArray | float:
    """Calculates the ppm difference between two values.

    Args:
        theoretical (float): Theoretical value
        observed (float): Observed value

    Returns:
        float: ppm difference
    """
    return (observed - theoretical) / theoretical * 10**6


def get_tolerance(
    tolerance: float = 25.0,
    theoretical: float | None = None,
    unit: MassError = "ppm",
) -> float:
    """Calculates the toleranc in daltons from either a dalton tolerance or a ppm.

    tolerance.

    Returns
    -------
        float, the tolerance value in daltons

    Examples
    --------
        >>> get_tolerance(25.0, 2000.0, "ppm")
        0.05
        >>> get_tolerance(0.02, 2000.0, "da")
        0.02
        >>> get_tolerance(25.0, np.array([1000.0, 1500.0, 2000.0]), "da")
        25.0
        >>> get_tolerance(25.0, np.array([1000.0, 1500.0, 2000.0]), "ppm")
        array([0.025 , 0.0375, 0.05  ])

    Args
    ----
      tolerance: Tolerance value to be used (Default value = 25.0)
      theoretical: Theoretical m/z to be used (only used for ppm)
      unit: Lietrally da for daltons or ppm for ... ppm (Default value = "ppm")
    """
    if unit == "ppm":
        if theoretical is None:
            raise ValueError("Theoretical m/z must be provided for ppm")
        return theoretical * tolerance / 10**6
    elif unit == "da":
        return tolerance
    else:
        raise ValueError(f"unit {unit} not implemented")


def is_in_tolerance(
    theoretical: float,
    observed: float,
    tolerance: float = 25.0,
    unit: MassError = "ppm",
) -> bool:
    """Checks wether an observed mass is close enough to a theoretical mass.

    Returns
    -------
        bool, Wether the value observed is within the tolerance of the theoretical value

    Examples
    --------
        >>> is_in_tolerance(2000.0, 2000.0, 25.0, "ppm")
        True
        >>> is_in_tolerance(2000.0, 2000.0, 25.0, "da")
        True
        >>> is_in_tolerance(2000.0, 2000.4, 25.0, "ppm")
        False
        >>> obs = np.array([1000.0, 1500.0, 2000.0])
        >>> theo = np.array([1000.001, 1500.001, 2000.2])
        >>> is_in_tolerance(theo, obs, 25.0, "ppm")
        array([ True,  True, False])

    Args
    ----
      theoretical: Theoretical m/z
      observed: Observed m/z
      tolerance: Tolerance value to be used (Default value = 25.0)
      unit: Lietrally da for daltons or ppm for ... ppm (Default value = "ppm")
    """
    mz_tolerance = get_tolerance(
        theoretical=theoretical, tolerance=tolerance, unit=unit
    )
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance

    return (lower <= theoretical) & (theoretical <= upper)


def binary_search_gte(arr, val):
    """Binary search for a value in an array.

    This variation finds the first element that is greater than the
    value provided.

    Args:
        arr (np.array): Array to search in
        val (float): Value to search for

    Returns:
        int: Index of the value
    """
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        # print(f"left: {arr[left]}, right: {arr[right]}, mid: {arr[mid]}, val: {val}")
        if arr[mid] == val:
            return mid
        elif arr[mid] < val:
            left = mid + 1
        else:
            right = mid - 1

    return left


def test_binary_gte():
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert binary_search_gte(arr, 1) == 0

    arr2 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    assert binary_search_gte(arr2, 1) == 0
    assert binary_search_gte(arr2, 2) == 0
    assert binary_search_gte(arr2, 10) == 3


def find_matching_sorted(
    A, B, max_diff=0.5, in_range_fun: Callable | None = None  # type: ignore
) -> tuple[NDArray, NDArray]:
    """Finds the matching between two sorted lists of floats.

    Args:
        A (List[float]): Sorted list of floats
        B (List[float]): Sorted list of floats
        max_diff (float, optional): Maximum allowed difference between
            elements in A and B. Defaults to 0.5.
        in_range_fun (Callable, optional): Function that takes two floats
            and returns True if they are in some definition of range.
            Defaults to None.

    Returns:
        List[Tuple[int, int], None, None]: Generator of tuples
    """

    if in_range_fun is None:

        def in_range_fun(x, y):
            return abs(x - y) <= max_diff

    elems = []
    min_ib = 0

    for ia, a in enumerate(A):
        min_val = a - max_diff
        min_ib = binary_search_gte(B, min_val)
        for ib, b in enumerate(B[min_ib:], start=min_ib):
            if b < min_val:
                min_ib = ib
                continue
            elif in_range_fun(a, b):
                elems.append((ia, ib))
            else:
                break

    if len(elems) == 0:
        ia, ib = [], []
    else:
        ia, ib = zip(*elems)
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def sort_all(keys, *args):
    """Sorts all the arrays in args by ordered version of the array in keys.

    Examples
    --------
        >>> keys = np.array([1, 3, 2, 4])
        >>> foo = np.array([1, 2, 3, 4])
        >>> bar = np.array([5, 6, 7, 8])
        >>> keys, foo, bar = sort_all(keys, foo, bar)
        >>> foo
        array([1, 3, 2, 4])
        >>> bar
        array([5, 7, 6, 8])

    Args
    ----
        keys: Array to be used as the key for sorting
        *args: Arrays to be sorted
    """

    index = np.argsort(keys, kind="mergesort")
    out_args = [keys] + [x for x in args]
    out = [np.array(x)[index] for x in out_args]
    return out


def find_matching(A, B, max_diff=0.5, in_range_fun=None) -> tuple[NDArray, NDArray]:
    """Finds the matching between two lists of floats.

    Args:
        A (List[float]): List of floats
        B (List[float]): List of floats
        max_diff (float, optional): Maximum allowed difference between
            elements in A and B. Defaults to 0.5.
        in_range_fun (Callable, optional): Function that takes two floats
            and returns True if they are in some definition of range.
            Defaults to None.

    Returns:
        np.array, np.array
    """

    a_indices = np.array(range(len(A)), dtype=int)
    b_indices = np.array(range(len(B)), dtype=int)

    A, a_indices = sort_all(A, a_indices)
    B, b_indices = sort_all(B, b_indices)

    ia, ib = find_matching_sorted(A, B, max_diff, in_range_fun)

    return a_indices[ia], b_indices[ib]


def annotate_peaks(
    theo_mz: NDArray,
    mz: NDArray,
    tolerance: float = 25.0,
    unit: MassError = "ppm",
) -> tuple[np.ndarray, np.ndarray]:
    """Annotate_peaks Assigns m/z peaks to annotations.

    Arguments:
        theo_mz {np.array}: Theoretical m/z values
        mz {np.array}: Observed m/z values
        tolerance {float}: Tolerance value to be used (Default value = 25.0)
        unit {MassError}: Lietrally da for daltons or ppm for ...
            ppm (Default value = "ppm")

    Returns:
        np.array, np.array:
            The two arrays are the same length to each other.
            They contain [1] the indices in the theoretical m/z array.
            that match [2] the indices in the observed m/z array.

            In other words ...
            z, w = annotate_peaks(x, y)
            peak x[z[i]] matches peak y[w[i]], being i any index.

    Examples:
        >>> dumm1 = np.array(
        ...     [
        ...         1500.0,
        ...         2000.0,
        ...         1000.0,
        ...     ]
        ... )
        >>> dumm2 = np.array([1000.001, 1600.001, 2000.2])
        >>> annotate_peaks(dumm1, dumm2, 1, "da")
        (array([2, 1]), array([0, 2]))
        >>> (
        ...     dumm1[annotate_peaks(dumm1, dumm2, 1, "da")[1]],
        ...     dumm2[annotate_peaks(dumm1, dumm2, 1, "da")[0]],
        ... )
        (array([1500., 1000.]), array([2000.2  , 1600.001]))
        >>> theo_dumm1 = (np.array(range(20)) * 0.01) + 1000
        >>> obs_dumm2 = np.array([1000.001, 1600.001, 2000.2])
        >>> annotate_peaks(theo_dumm1, obs_dumm2, 1, "da")
        (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
           13, 14, 15, 16, 17, 18, 19]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        >>> theoretical_peaks = np.array([100.0, 200.0, 300.0, 500.0])
        >>> theoretical_labels = np.array(["a", "b", "c", "d"])
        >>> mzs = np.array([10, 100.0, 100.001, 150.0, 200.0, 300.0, 400.0])
        >>> ints = np.array([0.1, 1.0, 0.2, 5.0, 31.0, 2.0, 3.0])
        >>> x, y = annotate_peaks(theoretical_peaks, mzs)
        >>> x
        array([0, 0, 1, 2])
        >>> y
        array([1, 2, 4, 5])
        >>> theoretical_labels[x]
        array(['a', 'a', 'b', 'c'], dtype='<U1')
        >>> ints[y]
        array([ 1. ,  0.2, 31. ,  2. ])
    """
    if len(theo_mz) == 0 or len(mz) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_delta = get_tolerance(tolerance=tolerance, theoretical=max(mz), unit=unit)

    def diff_fun(theo, obs):
        tol = get_tolerance(tolerance=tolerance, theoretical=theo, unit=unit)
        return abs(theo - obs) <= tol

    return find_matching(theo_mz, mz, max_delta, diff_fun)


def allign_intensities(
    mz1, mz2, int1, int2, tolerance: float = 25.0, unit: MassError = "ppm"
):
    """allign_intensities

    Arguments:
        mz1 {np.array}: Array of m/z values
        mz2 {np.array}: Array of m/z values
        int1 {np.array}: Intensities for mz1
        int2 {np.array}: Intensities for mz2
        tolerance {float}: Tolerance value to be used (Default value = 25.0)
        unit {str}:
           Lietrally da for daltons or ppm for ppm to use to calculate the mass errors
           (Default value = "ppm")

    Returns:
        np.array, np.array:
            The two arrays are the same length to each other.
            They contain the matched intensities between the input mz arrays.
    """
    x, y = annotate_peaks(mz1, mz2, tolerance, unit)
    int1_matched, int1_unmatched = int1[x], np.delete(int1, x)
    int2_matched, int2_unmatched = int2[y], np.delete(int2, y)

    int1_out = np.concatenate(
        [int1_matched, int1_unmatched, np.zeros_like(int2_unmatched)]
    )
    int2_out = np.concatenate(
        [int2_matched, np.zeros_like(int1_unmatched), int2_unmatched]
    )

    return int1_out, int2_out


def lazy(func):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = f"_lazy_{func.__name__}"

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


def clear_lazy_cache(self):
    """Clears the lazy cache of the object.

    This method is intended to be used when a modification is made
    in-place on classes that use the @lazy decorator for properties.
    """
    for attr in dir(self):
        if attr.startswith("_lazy_"):
            delattr(self, attr)
