import warnings
from typing import Callable, Dict, List, Union

import numpy as np


def pad_to_shape(x, shape):
    """
    Pad a numpy array to a given shape.

    Examples:
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pad_to_shape(x, (3, 3))
    array([[1, 2, 3],
           [4, 5, 6],
           [0, 0, 0]])
    """
    if x.shape == shape:
        return x
    elif x.shape[0] > shape[0]:
        error_msg = f"Cannot pad to shape {shape} because"
        error_msg += f" the first dimension is smaller than {x.shape[0]}"

        raise ValueError(error_msg)
    else:
        padding_arg = [(0, shape[i] - x.shape[i]) for i in range(len(shape))]
        pad = np.pad(x, padding_arg, mode="constant", constant_values=0)
        return pad


def default_collate(x):
    """
    Collate function for the default adapter.

    Examples:
    >>> x = [{'aa': np.array([[1, 2, 3], [4, 5, 6]]),
    ...      'mods': np.array([[1, 2, 3], [4, 5, 6]])},
    ...      {'aa': np.array([[1, 2, 3], [4, 5, 6]]),
    ...      'mods': np.array([[1, 2, 3], [4, 5, 6]])}]
    >>> {k: v.shape for k, v in x[0].items()}
    {'aa': (2, 3), 'mods': (2, 3)}
    >>> out = default_collate(x)
    >>> out
    {'aa': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]]), 'mods': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]])}
    >>> {k: v.shape for k, v in out.items()}
    {'aa': (2, 2, 3), 'mods': (2, 2, 3)}
    """
    elem = x[0]
    elem_type = type(elem)

    if isinstance(elem, np.ndarray):
        shapes = set([e.shape for e in x])
        if len(shapes) != 1:
            max_shape = tuple(max(s) for s in zip(*shapes))
            warnings.warn(
                f"Padding to shape {max_shape} because the shapes are not the same"
            )
            # TODO: consider wether to keep this as a warn or make it an error ...
            x = [pad_to_shape(e, max_shape) for e in x]

        return np.stack(x)
    elif isinstance(elem, (int, float)):
        return np.array(x)
    elif isinstance(elem, str):
        return np.array(x)
    elif isinstance(elem, (tuple, list)):
        return [default_collate(samples) for samples in zip(*x)]
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in x]) for key in elem}
    else:
        raise TypeError(f"default_collate found invalid type: {elem_type}")


def hook_collate(x, hooks: Union[Callable, List[Callable], Dict[str, Callable]]):
    """
    Collate function that applies functions to elements.

    Examples:
    >>> x = [{'aa': np.array([[1, 2, 3], [4, 5, 6]]),
    ...      'mods': np.array([[1, 2, 3], [4, 5, 6]])},
    ...      {'aa': np.array([[1, 2, 3], [4, 5, 6]]),
    ...      'mods': np.array([[1, 2, 3], [4, 5, 6]])}]
    >>> out = default_collate(x)
    >>> out
    {'aa': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]]), 'mods': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]])}
    >>> hook_collate(x, lambda x: x+1)
        {'aa': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]]), 'mods': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]])}
    >>> hook_collate(x, {'aa': lambda x: x+1, 'mods': lambda x: x+10})
    {'aa': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]]), 'mods': array([[[11, 12, 13],
            [14, 15, 16]],
           [[11, 12, 13],
            [14, 15, 16]]])}
    """
    elem = x[0]
    elem_type = type(elem)

    if not isinstance(hooks, elem_type) and not isinstance(hooks, Callable):
        error_msg = "Cannot use hook_collate with elements "
        error_msg += f"type {elem_type} and hooks type {type(hooks)}."
        error_msg += " The provided hooks should be either a single callable, "
        error_msg += "a list of callables, or a dictionary of callables whose "
        error_msg += "keys match the keys of the elements."

        raise ValueError(error_msg)

    if isinstance(elem, np.ndarray):
        x = [hooks(e) for e in x]
        shapes = set([e.shape for e in x])
        if len(shapes) != 1:
            raise ValueError("All arrays must have the same shape")
        else:
            return np.stack(x)
    elif isinstance(elem, (int, float)):
        x = [hooks(e) for e in x]
        return np.array(x)
    elif isinstance(elem, str):
        x = [hooks(e) for e in x]
        return np.array(x)
    elif isinstance(elem, (tuple, list)):
        hooks = [hooks for _ in len(elem)] if isinstance(hooks, Callable) else hooks
        return [[h(e) for e in samples] for h, samples in zip(hooks, zip(*x))]
    elif isinstance(elem, dict):
        hooks = {k: hooks for k in elem} if isinstance(hooks, Callable) else hooks
        return {
            key: hook_collate([d[key] for d in x], hooks=hooks[key]) for key in elem
        }
    else:
        raise TypeError(f"default_collate found invalid type: {elem_type}")
