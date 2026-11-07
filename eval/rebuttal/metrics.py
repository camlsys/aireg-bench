import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

LABELS = [1, 2, 3, 4, 5]


def kappa_sklearn(y, x):
    return float(cohen_kappa_score(np.asarray(x, int), np.asarray(y, int), weights="quadratic", labels=LABELS))


def kappa(y, x):
    """Quadratically weighted kappa written as 1 - observed/expected squared disagreement.

    Identical to sklearn's quadratic kappa for integer ratings, and also defined when the
    reference takes half-point values (e.g. the median of two annotators).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = np.mean((x - y) ** 2)
    exp = np.mean((x[:, None] - y[None, :]) ** 2)
    return float(1 - obs / exp)


def rho(y, x):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y)[0])


def bias(y, x):
    return float(np.mean(np.asarray(x, float) - np.asarray(y, float)))


def mae(y, x):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


METRICS = {"kappa_w": kappa, "rho": rho, "bias": bias, "mae": mae}


def all_metrics(ref, pred):
    return {k: f(ref, pred) for k, f in METRICS.items()}


def median_int(H):
    """Median over annotators, truncated to int as eval.py does (only matters for even counts)."""
    return np.nanmedian(H, axis=0).astype(int)
