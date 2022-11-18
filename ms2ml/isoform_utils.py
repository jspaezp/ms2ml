"""
This module is meant to be for internal consumption
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable
from itertools import product


class _unique_element:
    """Part of the answer from https://stackoverflow.com/questions/6284396."""

    def __init__(self, value, occurrences):
        self.value = value
        self.occurrences = occurrences


def perm_unique(elements: Iterable) -> Generator:
    """
    Perm_unique Gets permutations of elements taking into account repeated.

    Part of the answer from https://stackoverflow.com/questions/6284396

    Permutes the elements passed but skips all permutations where elements are
    the same. For instance (0, 1, 0) would give 3 possibilities.

    Parameters
    ----------
    elements : List or str
        Elements to be permuted

    Returns
    -------
    Generator
        A list with all permutations

    Examples
    --------
    >>> out = list(perm_unique("COM"))
    >>> sorted(out)
    [('C', 'M', 'O'), ('C', 'O', 'M'), ('M', 'C', 'O'),
     ('M', 'O', 'C'), ('O', 'C', 'M'), ('O', 'M', 'C')]
    >>> out = list(perm_unique("CCM"))
    >>> sorted(out)
    [('C', 'C', 'M'), ('C', 'M', 'C'), ('M', 'C', 'C')]
    >>> out = list(perm_unique([0, 1, 0]))
    >>> sorted(out)
    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    """
    eset = set(elements)
    listunique = [_unique_element(i, elements.count(i)) for i in eset]
    u = len(elements)
    return _perm_unique_helper(listunique, [0] * u, u - 1)


def _perm_unique_helper(listunique, result_list, d):
    """Part of the answer from https://stackoverflow.com/questions/6284396."""
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d] = i.value
                i.occurrences -= 1
                yield from _perm_unique_helper(listunique, result_list, d - 1)
                i.occurrences += 1


class LocationPlaceholder:
    def __init__(self, value):
        self.value = value

    def format(self, addition):
        return (self.value, addition)


def replace_placeholders(lst: list, replacements: list):
    replacements = list(replacements)
    out = []
    for x in lst:
        if isinstance(x, LocationPlaceholder):
            out.append(x.format(replacements.pop(0)))
        else:
            out.append(x)

    return tuple(out)


def get_elem_uniq_perm(seq_elems, aas, mod):
    mod_sampler = [x[1] for x in seq_elems if x[0] in aas]
    perm_iter = list(perm_unique(mod_sampler))

    return perm_iter


def get_elem_comb(seq_elems, aas, mod):
    num_mods_sites = sum([1 for x in seq_elems if x[0] in aas])
    out = list(product([None, mod], repeat=num_mods_sites))
    return out


def _prep_sequence(seq_elems):
    out = []

    for x in seq_elems:
        if not isinstance(x[0], str):
            raise TypeError(f"Sequence element {x} is not a string")
        if isinstance(x[1], str) or (x[1] is None):
            out.append((x[0], x[1]))
        elif isinstance(x[1], list) or isinstance(x[1], tuple):
            if len(x[1]) == 0:
                raise ValueError("Empty list of modifications")
            elif len(x[1]) > 1:
                raise ValueError(
                    "More than one modification per positions is not"
                    " supported at this time"
                )
            out.append((x[0], x[1][0]))
        else:
            raise NotImplementedError(
                f"Sequence element {x} -> {x[1]} {type(x[1])} is not supported"
            )

    return tuple(out)


def seq_comb_getter_factory(comb_getter):
    def wrapped(
        seq_elems: list[tuple[str, list[str]]], mod: str, aas: str
    ) -> list[str]:
        seq_elems = _prep_sequence(seq_elems)
        placeholder_seq = [
            x if (x[0] not in aas) else LocationPlaceholder(x[0]) for x in seq_elems
        ]

        out_seqs = []
        combs = comb_getter(seq_elems, aas, mod)
        for x in combs:
            out_seqs.append(replace_placeholders(placeholder_seq, x))

        return tuple(set(tuple(out_seqs)))

    return wrapped


_get_mod_isoforms = seq_comb_getter_factory(get_elem_uniq_perm)
_get_variable_mods = seq_comb_getter_factory(get_elem_comb)


def _mod_comb_getter_factory(comb_getter):
    def wrapped(seq_elems: str, variable_mods) -> list[str]:
        seq_elems = _prep_sequence(seq_elems)
        seqs = [seq_elems]

        for mod, aas in variable_mods.items():
            tmp_seqs = []
            for s in seqs:
                x = comb_getter(s, mod, aas)
                tmp_seqs.extend(list(set(x)))
                if len(tmp_seqs) > 10000:
                    warnings.warn(
                        "Large number of mod combinations found, clipping at 1k"
                    )
                    continue

            seqs.extend(tmp_seqs)

        out = list(set(seqs))
        return out

    return wrapped


def get_mod_isoforms(*args, **kwargs):
    """Gets modification isoforms for a peptide with modifications

    Example
    -------
    >>> seq_elems = [
    ...     ("n_term", None),
    ...     ("S", None),
    ...     ("M", None),
    ...     ("M", None),
    ...     ("S", ["[U:21]"]),
    ...     ("C", ["[U:4]"]),
    ...     ("c_term", None),
    ... ]
    >>> mods_list = {"[U:21]": "STY", "[U:35]": "M"}
    >>> out = list(get_mod_isoforms(seq_elems, mods_list))
    >>> out
    [...]
    >>> len(out)
    2
    """
    return _mod_comb_getter_factory(_get_mod_isoforms)(*args, **kwargs)


def get_mod_possible(*args, **kwargs):
    """Gets modification combinations for a peptide

    Example
    -------
    >>> seq_elems = [
    ...     ("n_term", None),
    ...     ("S", None),
    ...     ("M", None),
    ...     ("M", None),
    ...     ("S", ["[U:21]"]),
    ...     ("C", ["[U:4]"]),
    ...     ("c_term", None),
    ... ]
    >>> mods_list = {"[U:21]": "STY", "[U:35]": "M"}
    >>> out = list(get_mod_possible(seq_elems, mods_list))
    >>> out
    [...]
    >>> len(out)
    16
    """
    return _mod_comb_getter_factory(_get_variable_mods)(*args, **kwargs)
