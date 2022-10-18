import numpy as np


def cosine_similarity(x, y, eps=1e-6):
    """Computes the cosine similarity between two vectors.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        eps (float): Epsilon value to avoid division by zero.
    Returns:
        float: Cosine similarity between x and y.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + eps)


def pearson_correlation(x, y):
    """Computes the Pearson correlation between two vectors.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
    Returns:
        float: Pearson correlation between x and y.
    """
    return np.corrcoef(x, y)[0, 1]


def spearman_correlation(x, y):
    """Computes the Spearman correlation
    between two vectors.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
    Returns:
        float: Spearman correlation
        between x and y.
    """
    return pearson_correlation(np.argsort(x), np.argsort(y))


def spectral_angle(x, y, eps=1e-6):
    """Computes the spectral angle between two vectors.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        eps (float): Epsilon value to avoid division by zero.
    Returns:
        float: Spectral angle between x and y.
    """
    return 1 - (2 * (np.arccos(cosine_similarity(x, y, eps)) / np.pi))
