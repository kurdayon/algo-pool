# lwlr — locally weighted linear regression

One small regression model that contains ordinary linear regression, a
locally fitted linear model and a nearest-neighbour style weighted average
as special cases. Two parameters, `alpha` and `rho`, move you between them.

## The idea

Linear regression and nearest-neighbour regression look very different, but
both predict a new point as a weighted sum of the training targets:

```
y_hat(x) = sum_i w_i(x) * y_i
```

The only difference is how the weights `w_i(x)` are chosen. Linear regression
uses weights that come out of a least-squares solve and involve every training
point. Nearest neighbours gives weight only to the closest points. Locally
weighted linear regression does both at once: it fits a linear model, but with
training points weighted by how close they are to the query point.

## The formula

For a query point `x` the prediction is

```
y_hat(x) = x_aug^T (A^T W(x) A + lambda D)^-1 A^T W(x) y
```

where

| symbol   | meaning                                                                        |
|----------|--------------------------------------------------------------------------------|
| `A`      | the training matrix `X` with a column of ones added in front (the bias column) |
| `x_aug`  | the query point with a leading 1, i.e. `[1, x]`                                |
| `W(x)`   | diagonal matrix of weights, `w_i = exp(-alpha * ||x_i - x||)`, normalised to sum to 1 |
| `D`      | `diag(0, 1, 1, ..., 1)`: the ridge penalty acts on the slopes only, never on the bias |
| `lambda` | `rho / (1 - rho)`, the ridge strength                                           |

The distance is plain Euclidean distance between the query point and each
training point. The bias column is added internally in both `fit` and
`predict`; do not add it to your `X` yourself.

## The two parameters

**`alpha` (locality), `alpha >= 0`.** Controls how fast the weight of a
training point falls off with distance. `alpha = 0` gives every point the same
weight. A larger `alpha` makes the fit depend on fewer, closer points. As
`alpha` goes to infinity only the single closest point matters.

**`rho` (mixing), `0 <= rho <= 1`.** Controls how much the slopes are
pulled towards zero. `rho = 0` means no penalty, a full local linear fit.
`rho = 1` means the slopes are forced to zero, so only the bias is left, and
the bias of a weighted least-squares fit is simply the weighted mean of the
targets. Values in between interpolate smoothly. Internally `rho` is mapped
to a ridge strength `lambda = rho / (1 - rho)`, which runs from 0 to infinity.

## What you get in the corners

| `alpha` | `rho` | model                                                          |
|---------|-------|----------------------------------------------------------------|
| 0       | 0     | ordinary least-squares linear regression                       |
| 0       | 1     | predict the mean of `y` everywhere                             |
| > 0     | 0     | locally weighted linear regression (a LOESS-style local linear fit) |
| > 0     | 1     | kernel-weighted average of `y` (a soft nearest neighbour; with large `alpha` it becomes 1-NN) |
| 0       | 0 < rho < 1 | ridge regression with the bias left unpenalised           |

Compared with plain nearest-neighbour averaging, the local linear fit
(`rho = 0`) is usually better at the edges of the data and wherever the target
has a slope, because a weighted average can only pull predictions towards the
neighbours' values while a local line can extrapolate the trend a little.

## Usage

```python
import numpy as np
from lwlr import LocallyWeightedRegressor

X = np.random.rand(500, 3)
y = np.sin(3 * X[:, 0]) + X[:, 1] - 0.5 * X[:, 2]

model = LocallyWeightedRegressor(alpha=5.0, rho=0.0)
model.fit(X, y)
y_hat = model.predict(X[:10])
```

The class follows the scikit-learn pattern: `fit(X, y)` and `predict(X)`,
plus `get_params` / `set_params`, so it can be used with scikit-learn's
`cross_val_score` and `GridSearchCV` to pick `alpha` and `rho`.

Only NumPy is required.

## Things to know

**Scale your features.** The weights come from Euclidean distance on the raw
features, so a feature measured in thousands dominates one measured in units.
Standardise `X` (for example with `StandardScaler`) before fitting unless the
features are already on comparable scales. The same `alpha` then means the
same thing for every column.

**Weights are normalised.** For each query point the weights are divided by
their sum. This matters because the ridge term is not scale-free: doubling all
weights is the same as halving `lambda`. Normalising keeps the meaning of `rho`
the same at every query point and for every `alpha`. Before normalising, the
smallest distance is subtracted from all distances so that a very large
`alpha` cannot underflow every weight to zero.

**The bias is never penalised.** `D` has a zero in the bias position. If the
bias were penalised too, the `rho = 1` limit would predict zero instead of the
weighted average.

**Singular systems are handled.** With `rho = 0` and a large `alpha`, so few
points carry weight that `A^T W A` can be singular. The model solves with
`numpy.linalg.lstsq`, which returns the minimum-norm solution instead of
failing. A small positive `rho` is a cleaner fix if you see noisy predictions
in that regime.

**Cost.** There is nothing to learn in `fit`; it only stores the data. Each
call to `predict` solves one `(d+1) x (d+1)` linear system per query point
after computing distances to all `n` training points, so prediction costs
about `O(m * n * d)` for `m` query points. This is fine for thousands of
training points; for very large training sets a neighbour index would be
needed to cut the distance computations.

## Files

- `lwlr.py` — the `LocallyWeightedRegressor` class.
- `README.md` — this file.
