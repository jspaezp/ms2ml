from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .constants import PROTON
from .types import MassError


def mz(mass: float, charge: int) -> float:
    return (mass + (charge * PROTON)) / charge


def get_tolerance(
    tolerance: float = 25.0,
    theoretical: Optional[float] = None,
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


def is_sorted(
    lst: Sequence,
    key: Callable = lambda x: x,
) -> bool:
    """Is_sorted Checks if a list is sorted.

    Returns
    -------
        bool: Wether at least 1 element is out of order

    Examples
    --------
        >>> is_sorted([1, 2, 3, 4])
        True
        >>> is_sorted([1, 2, 2, 3])
        True
        >>> is_sorted([4, 2, 2, 3])
        False

    Args
    ----
        lst (List): List to check if it is sorted
        key (Callable, optional):
            Function to use as the key to compare.
            Defaults to lambda x:x.
    """
    for i, el in enumerate(lst[1:]):
        if key(el) < key(lst[i]):  # i is the index of the previous element
            return False
    return True


def sort_if_needed(
    lst: list,
    key: Callable = lambda x: x,
) -> list:
    """Sorts a list IN PLACE if it is not already sorted.

    Examples
    --------
    >>> foo = [1, 16, 3, 4]
    >>> _ = sort_if_needed(foo)
    >>> foo
    [1, 3, 4, 16]
    >>> foo = [[1, "A"], [16, "B"], [3, "C"], [4, "D"]]
    >>> _ = sort_if_needed(foo, key=lambda x: x[0])
    >>> foo
    [[1, 'A'], [3, 'C'], [4, 'D'], [16, 'B']]
    >>> foo = np.array([1, 16, 3, 4])
    >>> # sort_if_needed(foo) # breaks for arrays

    Args
    ----
        lst : List to be sorted
        key (Callable, optional): Function to use as the key for sorting.
            Defaults to lambda x:x.
    """

    # TODO benchmark if this is faster than just sorting
    if not is_sorted(lst, key):
        lst.sort(key=key)

    return lst


def sort_all(keys, *args):
    """Sorts all the arrays in args by ordered version of the array in keys.

    Examples
    --------
        >>> keys = np.array([1, 3, 2, 4])
        >>> foo = np.array([1, 2, 3, 4])
        >>> bar = np.array([5, 6, 7, 8])
        >>> foo, bar = sort_all(keys, foo, bar)
        >>> foo
        array([1, 3, 2, 4])
        >>> bar
        array([5, 7, 6, 8])

    Args
    ----
        keys: Array to be used as the key for sorting
        *args: Arrays to be sorted
    """

    index = np.argsort(keys)

    # TODO make error message for the case where this fails
    # due to a list being passed ...
    return (x[index] for x in args)


def annotate_peaks(
    theo_mz,
    mz,
    tolerance: float = 25.0,
    unit: MassError = "ppm",
) -> Tuple[np.ndarray, np.ndarray]:
    """Annotate_peaks Assigns m/z peaks to annotations.

    Returns
    -------
    Dict[str, float]:
        A dictionary with the keys being the names of the ions and the values being
        the intensities that were asigned to such ion.

    Examples
    --------
    >>> theoretical_peaks = np.array([100.0, 200.0, 300.0, 500.0])
    >>> theoretical_labels = np.array(["a", "b", "c", "d"])
    >>> mzs = np.array([10, 100.0, 100.001, 150.0, 200.0, 300.0, 400.0])
    >>> ints = np.array([0.1, 1.0, 0.2, 5.0, 31.0, 2.0, 3.0])
    >>> x, y = annotate_peaks(theoretical_peaks, mzs)
    >>> x
    array([1, 2, 4, 5])
    >>> y
    array([0, 0, 1, 2])
    >>> theoretical_labels[y]
    array(['a', 'a', 'b', 'c'], dtype='<U1')
    >>> ints[x]
    array([ 1. ,  0.2, 31. ,  2. ])

    Args
    ----
    theoretical_peaks:
        Dictionary specifying the names and masses of theoretical peaks
    mzs:
        Array of the masses to be annotated.
    tolerance:
        Tolerance to be used to count an observed and a theoretical m/z as a match.
        Defaults to 25.
    unit:
        The unit of the formerly specified tolerance (da or ppm).
        Defaults to "ppm".
    """
    max_delta = get_tolerance(tolerance=tolerance, theoretical=max(mz), unit=unit)

    mzs, mz_idx = sort_all(mz, mz, np.array(range(len(mz))))
    theo_peaks, theo_idx = sort_all(theo_mz, theo_mz, np.array(range(len(theo_mz))))

    mz_pairs = zip(mzs, mz_idx)
    theo_peaks = zip(theo_peaks, theo_idx)

    curr_theo_val, curr_theo_idx = next(theo_peaks)

    mz_indices = []
    annot_indices = []

    for mz, idx in mz_pairs:
        deltamass = mz - curr_theo_val
        try:
            # Skip values that cannot match the current theoretical peak
            while deltamass >= max_delta:
                curr_theo_val, curr_theo_idx = next(theo_peaks)
                deltamass = mz - curr_theo_val
        except StopIteration:
            pass

        in_deltam = abs(deltamass) <= max_delta
        if in_deltam and abs(deltamass) <= get_tolerance(
            curr_theo_val, tolerance, unit
        ):
            mz_indices.append(idx)
            annot_indices.append(curr_theo_idx)
    else:
        try:
            while True:
                curr_theo_val, curr_theo_idx = next(theo_peaks)
                deltamass = mz - curr_theo_val
                if deltamass < -max_delta:
                    break
                in_deltam = abs(deltamass) <= max_delta
                if in_deltam and abs(deltamass) <= get_tolerance(
                    curr_theo_val, tolerance, unit
                ):
                    mz_indices.append(idx)
                    annot_indices.append(curr_theo_idx)
        except StopIteration:
            pass

    # Remember to handle normalization prior to annotation
    # max_int = max([v for v in annots.values()] + [0])
    # annots = {k: v / max_int for k, v in annots.items()}

    # TODO benchmark if checking the tolerance directly
    # is faster than checking the delta mass

    # The first element is the indices of the m/z values that were annotated
    # The second element is the indices of the fragment values that matched the mzs
    return np.array(mz_indices), np.array(annot_indices)
