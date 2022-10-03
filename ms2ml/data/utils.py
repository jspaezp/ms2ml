import warnings
from typing import Callable, Dict, List, Union

import numpy as np


# zip taking strict=true is not supported in python 3.9
# pre Py3.10
def _strictzip(*args, strict=True):
    if strict:
        lengths = [len(y) for y in args]
        if not all(length == lengths[0] for length in lengths):
            err = f"zip() argument {lengths.index(min(lengths))}"
            err += f" is shorter than argument {lengths.index(max(lengths))}"
            raise ValueError(err)

    return zip(*args)


def _is_nested_numeric_list(x):
    if isinstance(x, (list, tuple)):
        return all(_is_nested_numeric_list(e) for e in x)
    elif isinstance(x, (int, float, np.number, np.ndarray)):
        return True
    else:
        return False


def _is_uniform_nested_list(x):
    if _is_nested_numeric_list(x):
        if isinstance(x, np.ndarray):
            return list(x.shape)

        if isinstance(x, (float, int)):
            return [1]

        elem, elem_type = _get_elem_info(x)
        if isinstance(elem, (list, tuple)):
            elem_shape = _is_uniform_nested_list(elem)
            if not elem_shape:
                return False

            elem_shape = tuple(elem_shape)

            for y in x:
                y_shape = _is_uniform_nested_list(y)
                if not y_shape:
                    return False
                y_shape = tuple(y_shape)
                if y_shape != elem_shape:
                    return False

            return [len(x)] + list(elem_shape)

        if isinstance(elem, np.ndarray):
            elem_shape = elem.shape
            for y in x:
                if isinstance(y, list):
                    breakpoint()
                if y.shape != elem_shape:
                    return False

            return [len(x)] + list(elem_shape)

        if isinstance(elem, (float, int)):
            return [len(x), 1]

    else:
        return False


def pad_to_shape(x, shape):
    """Pad a numpy array to a given shape.

    Examples
    --------
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


def pad_to_max_shape(x):
    shapes = {e.shape for e in x}
    if len(shapes) != 1:
        try:
            max_shape = tuple(max(s) for s in _strictzip(*shapes))
        except ValueError as e:
            if "zip() argument" in str(e) and "than argument" in str(e):
                error_msg = "Could not get the maximum length of the dimensions"
                error_msg += " in the batch when trying to pad to the same shapes,"
                error_msg += f" they have shapes: {shapes}."
                error_msg += " Maybe you missed nesting the elements in a list?"

                raise ValueError(error_msg)
            else:
                raise

        warnings.warn(
            f"Padding to shape {max_shape} because the shapes are not the same"
        )
        # TODO: consider wether to keep this as a warn or make it an error ...
        x = [pad_to_shape(e, max_shape) for e in x]

    return x


def _get_elem_info(x):
    try:
        type(x)
        elem = x[0]
    except KeyError as e:
        error_msg = "Cannot get the elements of the batch at position"
        error_msg += f" 0: {e}, Try to wrap the elements in a list"
        error_msg += " (`default_collate([...])`)"

        raise KeyError(error_msg)

    elem_type = type(elem)
    return elem, elem_type


def _check_list_lengths(*args, context=""):
    lengths = {len(y) for y in args}

    if len(lengths) != 1:
        error_msg = (
            f"The lengths of the lists are not the same, found lengths: {lengths}"
        )
        error_msg += " Maybe you missed nesting the elements in a list?"
        if context:
            error_msg += f" Context: {context}"

        raise ValueError(error_msg)


def _transpose_list(x):
    try:
        x = list(_strictzip(*x, strict=True))
    except ValueError:
        _check_list_lengths(*x, context="when trying to transpose the list")
        raise

    return x


def _default_collate(x, pad_shapes, level):
    elem, elem_type = _get_elem_info(x)

    if isinstance(elem, np.ndarray):
        if pad_shapes:
            # TODO improve error message to suggest element padding
            # and pointing at an example of the bad elems
            x = pad_to_max_shape(x)
        return np.stack(x, axis=0)
    elif isinstance(elem, (int, float, np.number)):
        return _default_collate([np.array(y) for y in x], pad_shapes, level + 1)
    elif isinstance(elem, str):
        return np.array(x)
    elif isinstance(elem, (tuple, list)):
        _check_list_lengths(
            *x, context="when trying to collate at nesting level " + str(level)
        )
        if level < 1:
            x = _transpose_list(x)

        if _is_uniform_nested_list(x):
            if level >= 1:
                # out = [np.stack(y) for y in _transpose_list(x)]
                x = np.stack(x)
                return x

        x = [_default_collate(s, pad_shapes=pad_shapes, level=level + 1) for s in x]
        out = [np.stack(e) if _is_uniform_nested_list(e) else e for e in x]
        out = np.stack(out) if _is_uniform_nested_list(out) else out
        return out
    elif isinstance(elem, dict):
        return {
            key: _default_collate(
                [d[key] for d in x], pad_shapes=pad_shapes, level=max(0, level - 1)
            )
            for key in elem
        }
    else:
        raise TypeError(f"default_collate found invalid type: {elem_type}")


def pad_collate(x):
    return _default_collate(x, pad_shapes=True, level=0)


def default_collate(x, pad_shapes=False):
    """Collate function for the default adapter.

    Examples
    --------
    >>> x = [
    ...     {
    ...         "aa": np.array([[1, 2, 3], [4, 5, 6]]),
    ...         "mods": np.array([[1, 2, 3], [4, 5, 6]]),
    ...     },
    ...     {
    ...         "aa": np.array([[1, 2, 3], [4, 5, 6]]),
    ...         "mods": np.array([[1, 2, 3], [4, 5, 6]]),
    ...     },
    ... ]
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
    return _default_collate(x, pad_shapes=pad_shapes, level=0)


def _zip_hooks(hooks, x):
    try:
        zipped = list(_strictzip(hooks, x, strict=True))
    except ValueError as err:
        zip_error = "zip() argument" in str(err)
        shorter_longer = ("is shorter than" in str(err)) or (
            "is longer than" in str(err)
        )
        zip_error = zip_error and shorter_longer
        if zip_error:
            error_msg = f"The number of hooks ({len(hooks)}) must match the number"
            error_msg += f" of elements in each batch ({len(x)})."
            raise ValueError(error_msg)
        else:
            raise

    return zipped


def _check_hooks(hooks, elem):
    compatible = False

    elem_description = f"element of type {type(elem)}"

    if isinstance(elem, dict):
        elem_description += f" and keys {elem.keys()}"
    elif isinstance(elem, (tuple, list)):
        elem_description += f" and length {len(elem)}"

    if isinstance(hooks, Callable):
        compatible = True
    elif isinstance(hooks, (list, tuple)):
        hook_description = f"hook of type {type(hooks)}, and length {len(hooks)}"
        if len(hooks) == len(elem):
            compatible = True
    elif isinstance(hooks, dict):
        hook_description = f"hook of type {type(hooks)}, and keys {hooks.keys()}"
        if isinstance(elem, dict):
            if all(x in hooks for x in elem):
                compatible = True
    else:
        hook_description = f"hook of type {type(hooks)}"

    if not compatible:
        error_msg = "Cannot use hook_collate with "
        error_msg += elem_description
        error_msg += " and " + hook_description
        error_msg += ". The provided hooks should be either a single callable, "
        error_msg += "a list of callables, or a dictionary of callables whose "
        error_msg += "keys match the keys of the elements."

        raise ValueError(error_msg)


def _hook_collate(x, hooks, pad_shapes=False, level=0):
    elem, elem_type = _get_elem_info(x)
    _check_hooks(hooks, elem)

    if isinstance(elem, np.ndarray):
        x = [hooks(y) for y in x]
        x = pad_to_max_shape(x)
        x = np.stack(x)
        return x
    elif isinstance(elem, (int, float)):
        x = [hooks(e) for e in x]
        x = np.stack(x)
        return x

    elif isinstance(elem, str):
        x = [hooks(e) for e in x]
        x = np.array(x)
        return x
    elif isinstance(elem, (tuple, list)):
        hooks = [hooks for _ in elem] if isinstance(hooks, Callable) else hooks
        _check_list_lengths(
            *x, context="when trying to collate at nesting level " + str(level)
        )

        x = _transpose_list(x)
        zipped = _zip_hooks(hooks, x)
        if _is_uniform_nested_list(x):
            if level >= 1:
                x = [[h(y) for y in samples] for h, samples in zipped]
                x = np.stack(_transpose_list(x))
                return x

        x = [
            _hook_collate(samples, hooks=h, pad_shapes=pad_shapes, level=level + 1)
            for h, samples in zipped
        ]

        out = [np.stack(e) if _is_uniform_nested_list(e) else e for e in x]
        out = np.stack(out) if _is_uniform_nested_list(out) else out
        return out

    elif isinstance(elem, dict):
        hooks = {k: hooks for k in elem} if isinstance(hooks, Callable) else hooks
        return {
            key: _hook_collate(
                [d[key] for d in x],
                hooks=hooks[key],
                pad_shapes=pad_shapes,
                level=max(0, level - 1),
            )
            for key in elem
        }
    else:
        raise TypeError(f"default_collate found invalid type: {elem_type}")


def hook_collate(
    x, hooks: Union[Callable, List[Callable], Dict[str, Callable]], pad_shapes=False
):
    """Collate function that applies functions to elements.

    Examples
    --------
    >>> x = [
    ...     {
    ...         "aa": np.array([[1, 2, 3], [4, 5, 6]]),
    ...         "mods": np.array([[1, 2, 3], [4, 5, 6]]),
    ...     },
    ...     {
    ...         "aa": np.array([[1, 2, 3], [4, 5, 6]]),
    ...         "mods": np.array([[1, 2, 3], [4, 5, 6]]),
    ...     },
    ... ]
    >>> out = default_collate(x)
    >>> out
    {'aa': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]]), 'mods': array([[[1, 2, 3],
            [4, 5, 6]],
           [[1, 2, 3],
            [4, 5, 6]]])}
    >>> hook_collate(x, lambda x: x + 1)
        {'aa': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]]), 'mods': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]])}
    >>> hook_collate(x, {"aa": lambda x: x + 1, "mods": lambda x: x + 10})
    {'aa': array([[[2, 3, 4],
            [5, 6, 7]],
           [[2, 3, 4],
            [5, 6, 7]]]), 'mods': array([[[11, 12, 13],
            [14, 15, 16]],
           [[11, 12, 13],
            [14, 15, 16]]])}
    """
    return _hook_collate(x, hooks=hooks, pad_shapes=pad_shapes, level=0)
