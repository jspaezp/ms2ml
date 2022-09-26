import numpy as np
import pytest
from numpy import allclose

from ms2ml.data.utils import (
    _check_hooks,
    _is_nested_numeric_list,
    _is_uniform_nested_list,
    default_collate,
    hook_collate,
)

hook_collate2 = pytest.param(lambda x: hook_collate(x, lambda x: x), id="hook_collate")
default_collate_pad = pytest.param(
    lambda x: default_collate(x, pad_shapes=True), id="default_collate_pad"
)  # noqa E731
default_collate_no_pad = pytest.param(
    lambda x: default_collate(x, pad_shapes=False), id="default_collate_no_pad"
)

as_is = lambda x: x  # noqa E731


def test_is_numeric_nested():
    assert _is_nested_numeric_list(1)
    assert _is_nested_numeric_list([1])
    assert _is_nested_numeric_list([[1]])
    assert _is_nested_numeric_list([[[[1]]], [1], [1, 1]])
    assert not _is_nested_numeric_list([[[[1]]], [1], [1, "a"]])


def test_is_uniform_nested_list():
    assert _is_uniform_nested_list([[1], [1]])
    assert not _is_uniform_nested_list([[1], [1, 1]])
    assert _is_uniform_nested_list([[1, 1], [1, 1]])
    assert _is_uniform_nested_list([[[1, 1]], [[1, 1]], [[1, 1]]])
    assert not _is_uniform_nested_list([[1], [1, 1], [1, 1, 1, 1]])
    assert not _is_uniform_nested_list([[1], [1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert not _is_uniform_nested_list(
        [[1], [1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    )


def test_hook_verification():
    # TODO: parametrize this and check error messages
    _check_hooks([as_is, as_is], [1, 1])
    _check_hooks([as_is, as_is], [[1], [1]])
    _check_hooks([as_is, as_is], {"a": 1, "b": 1})
    _check_hooks({"a": as_is, "b": as_is}, {"a": [1], "b": [1]})
    _check_hooks(as_is, {"a": [[1]], "b": [[1]]})

    with pytest.raises(ValueError):
        _check_hooks([as_is] * 3, {"a": [[1]], "b": [[1]]})

    with pytest.raises(ValueError):
        _check_hooks([as_is], {"a": [[1]], "b": [[1]]})

    with pytest.raises(ValueError):
        _check_hooks([as_is, as_is], [[[1]]])

    with pytest.raises(ValueError):
        _check_hooks([as_is, as_is], [[[[1]]]])


@pytest.mark.parametrize("collate_fn", [default_collate_pad, hook_collate2])
@pytest.mark.parametrize(
    "array_fun,error_strs",
    [
        (
            np.array,
            [
                "get the maximum length",
                "dimensions in the batch",
                "they have shapes",
                "Maybe you missed",
            ],
        ),
        pytest.param(
            as_is,
            [
                "The lengths of the list",
                "found lengths",
                "Maybe you missed",
            ],
            id="as_is",
        ),
    ],
)
def test_collate_error(collate_fn, array_fun, error_strs):
    good_batch = [array_fun([1]), array_fun([2, 3])]
    baad_batch = [array_fun([4]), array_fun([[5, 6]])]
    with pytest.raises(ValueError) as exc_info:
        collate_fn([good_batch, baad_batch])

    for error_str in error_strs:
        assert error_str in str(exc_info.value)


@pytest.mark.parametrize("collate_fn", [default_collate, hook_collate2])
@pytest.mark.parametrize("array_fun", [np.array, as_is])
def test_collate_list(collate_fn, array_fun):
    batch1 = [array_fun([1]), array_fun([2, 3])]
    batch2 = [array_fun([4]), array_fun([5, 6])]
    batch3 = [array_fun([7]), array_fun([8, 9])]

    batch = collate_fn([batch1, batch2, batch3])
    assert len(batch) == 2
    assert batch[0].shape == (
        3,
        1,
    )
    assert batch[1].shape == (
        3,
        2,
    )

    assert allclose(batch[0], np.array([[1], [4], [7]]))


# @pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize("collate_fn", [default_collate, hook_collate2])
def test_default_0d(collate_fn):
    array_fun = as_is
    batch1 = [1, 2, 3]
    batch1 = [array_fun(x) for x in batch1]

    batch = collate_fn(batch1)

    assert len(batch) == 3
    assert batch.shape == (3,)
    assert batch[0] == 1
    assert batch[1] == 2
    assert batch[2] == 3

    batch = collate_fn([batch1])

    # assert isinstance(batch, list)
    assert len(batch) == 3
    assert len(batch[0]) == 1
    assert batch[0].shape == (1,)
    assert batch[0][0] == 1
    assert batch[1][0] == 2
    assert batch[2][0] == 3

    batch2 = [4, 5, 6]
    batch2 = [array_fun(x) for x in batch2]
    batch = collate_fn([batch1, batch2])

    assert len(batch[0]) == 2
    assert allclose(batch[0], [1, 4])
    assert allclose(batch[1], [2, 5])


# @pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize(
    "collate_fn", [default_collate_pad, default_collate_no_pad, hook_collate2]
)
def test_default_collate_0d_dict(collate_fn):
    array_fun = as_is
    batch1 = {"a": [1, 2, 3], "b": [4, 5]}
    batch1 = {k: array_fun(v) for k, v in batch1.items()}
    batch = collate_fn([batch1])

    assert len(batch) == 2
    assert len(batch["a"]) == 3
    assert allclose(batch["a"][0], [1])

    batch2 = {"a": [7, 8, 9], "b": [10, 11]}
    batch2 = {k: array_fun(v) for k, v in batch2.items()}
    batch = collate_fn([batch1, batch2])

    assert len(batch) == 2
    assert allclose(batch["a"][0], [1, 7])
    assert allclose(batch["a"][1], [2, 8])
    assert allclose(batch["b"][0], [4, 10])
    assert allclose(batch["b"][1], [5, 11])


@pytest.mark.parametrize(
    "collate_fn", [default_collate_pad, default_collate_no_pad, hook_collate2]
)
@pytest.mark.parametrize("array_fun", [np.array, as_is])
def test_default_collate_dict(collate_fn, array_fun):
    batch1 = {"a": [[1, 2, 3]], "b": [[4, 5]]}
    batch1 = {k: [array_fun(x) for x in v] for k, v in batch1.items()}
    batch = collate_fn([batch1])

    assert len(batch) == 2
    assert len(batch["a"][0]) == 1
    assert allclose(batch["a"][0], [1, 2, 3])

    batch2 = {"a": [[7, 8, 9]], "b": [[10, 11]]}
    batch2 = {k: [array_fun(x) for x in v] for k, v in batch2.items()}
    batch = collate_fn([batch1, batch2])

    assert len(batch) == 2
    assert batch["a"].shape == (1, 2, 3)  # SHould this be an array?
    # assert batch["a"].shape == (1, 2, 3)
    assert allclose(batch["a"][0][0], [1, 2, 3])
    assert allclose(batch["a"][0][1], [7, 8, 9])
    assert allclose(batch["b"][0][0], [4, 5])
    assert allclose(batch["b"][0][1], [10, 11])


@pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize("hooks", [lambda x: x + 1, lambda x: x * 2])
def test_hook_collate(array_fun, hooks):
    batch = [1, 2, 3]
    batch = [array_fun(x) for x in batch]
    batch = hook_collate(batch, hooks)

    results = [1, 2, 3]
    results = [hooks(x) for x in results]
    assert allclose(batch, results)


@pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize("hooks", [lambda x: x + 1, lambda x: x * 2])
def test_hook_collate_list(array_fun, hooks):
    batch1 = [1, 2, 3]
    batch1 = [array_fun(x) for x in batch1]

    batch2 = [4, 5, 6]
    batch2 = [array_fun(x) for x in batch2]
    batch = hook_collate([batch1, batch2], hooks)

    results1 = [1, 4]
    results1 = [hooks(x) for x in results1]
    results2 = [2, 5]
    results2 = [hooks(x) for x in results2]
    results3 = [3, 6]
    results3 = [hooks(x) for x in results3]

    assert allclose(batch[0], results1)
    assert allclose(batch[1], results2)
    assert allclose(batch[2], results3)


@pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize(
    "hooks", [[lambda x: x + 1, lambda x: x * 2, lambda x: x**2]]
)
def test_hook_collate_list_multihook(array_fun, hooks):
    batch1 = [1, 2, 3]
    batch1 = [array_fun(x) for x in batch1]

    batch2 = [4, 5, 6]
    batch2 = [array_fun(x) for x in batch2]
    batch = hook_collate([batch1, batch2], hooks)

    results1 = [1, 4]
    results1 = [hooks[0](x) for x in results1]
    results2 = [2, 5]
    results2 = [hooks[1](x) for x in results2]
    results3 = [3, 6]
    results3 = [hooks[2](x) for x in results3]

    assert allclose(batch[0], results1)
    assert allclose(batch[1], results2)
    assert allclose(batch[2], results3)


@pytest.mark.parametrize("array_fun", [np.array, as_is])
def test_hook_collate_list_multihook_error(array_fun):
    batch1 = [1, 2, 3]
    batch1 = [array_fun(x) for x in batch1]

    batch2 = [4, 5, 6]
    batch2 = [array_fun(x) for x in batch2]

    hooks = [lambda x: x + 1, lambda x: x * 2, lambda x: x**2, lambda x: x**3]

    with pytest.raises(ValueError):
        hook_collate([batch1, batch2], hooks)

    hooks = [lambda x: x + 1, lambda x: x * 2]
    with pytest.raises(ValueError):
        hook_collate([batch1, batch2], hooks)


@pytest.mark.parametrize("array_fun", [np.array, as_is])
@pytest.mark.parametrize("hooks", [lambda x: x + 1, lambda x: x * 2])
def n_test_hook_collate_dict(array_fun, hooks):
    raise NotImplementedError
    batch1 = {"a": [[1, 2, 3]], "b": [[4, 5]]}
    batch1 = {k: [array_fun(x) for x in v] for k, v in batch1.items()}
    batch = hook_collate([batch1])

    assert len(batch) == 2
    assert len(batch["a"][0]) == 1
    results_1 = [1, 2, 3]
    results_1 = [hooks(x) for x in results_1]
    assert allclose(batch["a"][0], results_1)

    batch2 = {"a": [[7, 8, 9]], "b": [[10, 11]]}
    batch2 = {k: [array_fun(x) for x in v] for k, v in batch2.items()}
    batch = hook_collate([batch1, batch2])

    assert len(batch) == 2
    assert len(batch["a"]) == 2
    assert allclose(batch["a"][0][0], [1, 2, 3])
    assert allclose(batch["a"][0][1], [7, 8, 9])
    assert allclose(batch["b"][0][0], [4, 5])
    assert allclose(batch["b"][0][1], [10, 11])
    raise NotImplementedError
