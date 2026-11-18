"""Locally weighted linear regression with a ridge penalty on the slopes.

For a query point x the prediction is

    y_hat(x) = x_aug^T (A^T W(x) A + lambda D)^-1 A^T W(x) y

where
    A        = [1, X]            training features with a bias column added
    x_aug    = [1, x]            query point with a bias entry added
    W(x)     = diag(w_1 .. w_n)  w_i = exp(-alpha * ||x_i - x||), normalised to sum 1
    D        = diag(0, 1, .., 1) penalises the slopes only, never the bias
    lambda   = rho / (1 - rho)   so rho in [0, 1] covers lambda in [0, inf]

Corner cases:
    alpha = 0, rho = 0   ordinary least squares
    alpha = 0, rho = 1   predict the mean of y
    alpha > 0, rho = 0   locally weighted linear regression
    alpha > 0, rho = 1   kernel weighted average of y (a soft nearest neighbour)
"""

import numpy as np


class LocallyWeightedRegressor:
    """Locally weighted linear regression with a slope-only ridge penalty.

    Parameters
    ----------
    alpha : float, default 0.0
        Locality. Weight of training point i for query x is
        exp(-alpha * euclidean_distance(x_i, x)). 0 means every point gets
        the same weight; larger values make the fit more local.
    rho : float, default 0.0
        Mixing between a local linear fit (0) and a plain weighted average
        of the targets (1). Internally lambda = rho / (1 - rho) is used as
        the ridge penalty on the slope coefficients. The bias is never
        penalised.

    Notes
    -----
    Distances are Euclidean on the raw features, so scale your features
    before fitting if they live on different scales.

    There is no training step beyond storing the data. All work happens in
    ``predict``, which solves one small linear system per query point.
    """

    def __init__(self, alpha=0.0, rho=0.0):
        self.alpha = alpha
        self.rho = rho

    # ------------------------------------------------------------------ #
    # scikit-learn style plumbing
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        return {"alpha": self.alpha, "rho": self.rho}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in ("alpha", "rho"):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------ #
    # fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be a 1D array with one entry per row of X")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")

        self.X_ = X
        self.y_ = y
        self.A_ = self._add_bias(X)
        n_features = X.shape[1]
        # Penalty matrix: zero for the bias, one for every slope.
        self.D_ = np.diag(np.r_[0.0, np.ones(n_features)])
        return self

    def predict(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"X must be 2D with {self.X_.shape[1]} columns, got shape {X.shape}"
            )
        return np.array([self._predict_one(x) for x in X])

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _weights(self, x):
        """Normalised weights of every training point for one query point."""
        dist = np.linalg.norm(self.X_ - x, axis=1)
        # Subtract the smallest distance before exponentiating so that a large
        # alpha cannot underflow every weight to zero. The normalisation
        # below removes the constant factor again.
        w = np.exp(-self.alpha * (dist - dist.min()))
        return w / w.sum()

    def _predict_one(self, x):
        w = self._weights(x)

        if self.rho >= 1.0:
            # lambda = infinity: slopes are zero, only the bias remains and
            # the bias is the weighted mean of the targets.
            return float(w @ self.y_)

        lam = self.rho / (1.0 - self.rho)
        AW = self.A_ * w[:, None]                 # A^T W as (n, p) rows scaled
        G = self.A_.T @ AW + lam * self.D_        # A^T W A + lambda D
        b = AW.T @ self.y_                        # A^T W y
        # lstsq instead of a plain inverse: with lambda = 0 and a very local
        # weighting, G can be singular. lstsq then gives the minimum-norm fit.
        beta = np.linalg.lstsq(G, b, rcond=None)[0]
        x_aug = np.r_[1.0, x]
        return float(x_aug @ beta)
