import numpy as np
import pytest

from ms2ml.metrics.base import (
    cosine_similarity,
    pearson_correlation,
    spearman_correlation,
    spectral_angle,
)

# >>> import torchvision.datasets as datasets
# >>> mnist_trainset = datasets.MNIST(root='~/data',
# ... train=True, download=True, transform=None)
# >>> samples = [np.array(mnist_trainset[i][0])[::4, ::4] for i in [1,3,40,21]]
mnist_zeros = [
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 159, 0, 0],
            [0, 0, 0, 252, 252, 253, 0],
            [0, 0, 178, 19, 0, 253, 0],
            [0, 0, 230, 0, 0, 253, 0],
            [0, 0, 249, 85, 223, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 63, 0],
            [0, 0, 0, 181, 251, 251, 0],
            [0, 0, 240, 42, 109, 197, 0],
            [0, 0, 251, 0, 0, 251, 0],
            [0, 0, 251, 72, 251, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    ),
]

mnist_ones = [
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 128, 0, 0],
            [0, 0, 0, 0, 147, 0, 0],
            [0, 0, 0, 24, 0, 0, 0],
            [0, 0, 0, 46, 0, 0, 0],
            [0, 0, 0, 87, 0, 0, 0],
        ],
        dtype=float,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 68, 31, 0],
            [0, 0, 0, 0, 251, 0, 0],
            [0, 0, 0, 251, 0, 0, 0],
            [0, 0, 0, 253, 0, 0, 0],
            [0, 0, 24, 0, 0, 0, 0],
        ],
        dtype=float,
    ),
]


cosine_similarity(mnist_zeros[0].flatten(), mnist_zeros[1].flatten())
cosine_similarity(mnist_zeros[0].flatten(), mnist_ones[1].flatten())


@pytest.mark.parametrize(
    "metric",
    [cosine_similarity, pearson_correlation, spearman_correlation, spectral_angle],
)
def test_metrics_work(metric):
    similar = [mnist_zeros[0].flatten(), mnist_zeros[1].flatten()]
    dissimilar = [mnist_zeros[0].flatten(), mnist_ones[1].flatten()]

    calc_metric_similar = metric(*similar)
    calc_metric_disimilar = metric(*dissimilar)

    assert isinstance(calc_metric_similar, float)
    assert isinstance(calc_metric_disimilar, float)

    assert calc_metric_similar > -1
    assert calc_metric_similar < 1
    assert calc_metric_disimilar > -1
    assert calc_metric_disimilar < 1

    assert calc_metric_similar > calc_metric_disimilar
