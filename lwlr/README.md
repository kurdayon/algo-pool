# lwlr — locally weighted linear regression

One small regression model that contains ordinary linear regression, a
locally fitted linear model and a nearest-neighbour style weighted average
as special cases. Two parameters, `alpha` and `rho`, move you between them.

## Two models with closed-form solutions

Suppose we have a training matrix $X \in \mathbb{R}^{n \times d}$ (one row per
sample, no column of ones) and a target vector $y \in \mathbb{R}^n$. Two
classic regression models need no iterative training at all; both have an
analytical solution.

**Linear regression.** Add a bias column of ones to get
$A = [\mathbf{1}, X] \in \mathbb{R}^{n \times (d+1)}$. The least-squares
coefficients are

$$
\hat\beta = (A^\top A)^{-1} A^\top y ,
$$

and the prediction for a query point $x$ (with $x_{\text{aug}} = [1, x]$) is

$$
\hat y(x) = x_{\text{aug}}^\top \hat\beta = x_{\text{aug}}^\top (A^\top A)^{-1} A^\top y .
$$

**Nearest-neighbour style weighted average.** Give every training point a
weight that depends only on its distance to the query point, and predict the
weighted mean of the targets:

$$
\hat y(x) = \frac{\sum_{i=1}^{n} w_i(x)\, y_i}{\sum_{i=1}^{n} w_i(x)},
\qquad
w_i(x) = \exp\!\left(-\alpha \,\lVert x_i - x \rVert\right).
$$

Plain $k$-nearest neighbours is the same formula with weights that are $1$
for the $k$ closest points and $0$ for all others.

Both models are *linear smoothers*: in both cases the prediction is a weighted
sum of the training targets, $\hat y(x) = \sum_i s_i(x)\, y_i$, where the
weights $s_i(x)$ depend on $X$ and $x$ but not on $y$. The only difference is
how the weights are built. That is what makes it possible to put the two models
into one formula.

## The combined formula

For a query point $x$ the model predicts

$$
\hat y(x) = x_{\text{aug}}^\top \left(A^\top W(x)\, A + \lambda D\right)^{-1} A^\top W(x)\, y ,
$$

where

| symbol | meaning |
|---|---|
| $A = [\mathbf{1}, X]$ | the training matrix with a column of ones added in front (the bias column) |
| $x_{\text{aug}} = [1, x]$ | the query point with a leading $1$ |
| $W(x) = \mathrm{diag}(w_1(x), \dots, w_n(x))$ | diagonal matrix of weights, $w_i(x) = \exp(-\alpha \lVert x_i - x\rVert)$, normalised so that $\sum_i w_i(x) = 1$ |
| $D = \mathrm{diag}(0, 1, 1, \dots, 1)$ | the ridge penalty acts on the slopes only, never on the bias |
| $\lambda = \dfrac{\rho}{1 - \rho}$ | the ridge strength, running from $0$ (at $\rho = 0$) to $\infty$ (at $\rho = 1$) |

The distance $\lVert x_i - x \rVert$ is the plain Euclidean distance between
the query point and each training point. The bias column is added internally
in both `fit` and `predict`; do not add it to your `X` yourself.

This is a weighted least-squares fit with a ridge penalty on the slopes. It
contains both closed-form models above: with equal weights and no penalty it is
linear regression, and when the penalty pushes the slopes to zero only the bias
survives, and the bias of a weighted least-squares fit is exactly the weighted
mean of the targets.

## The two parameters

**`alpha` (locality), $\alpha \ge 0$.** Controls how fast the weight of a
training point falls off with distance. $\alpha = 0$ gives every point the same
weight. A larger $\alpha$ makes the fit depend on fewer, closer points. As
$\alpha \to \infty$ only the single closest point matters.

**`rho` (mixing), $0 \le \rho \le 1$.** Controls how much the slopes are
pulled towards zero. $\rho = 0$ means no penalty, a full local linear fit.
$\rho = 1$ means the slopes are forced to zero, so only the bias is left, which
is the weighted mean of the targets. Values in between interpolate smoothly.
Internally $\rho$ is mapped to the ridge strength $\lambda = \rho / (1 - \rho)$.

## What you get in the corners

| `alpha` | `rho` | model |
|---|---|---|
| $0$ | $0$ | ordinary least-squares linear regression |
| $0$ | $1$ | predict the mean of $y$ everywhere |
| $> 0$ | $0$ | locally weighted linear regression (a LOESS-style local linear fit) |
| $> 0$ | $1$ | kernel-weighted average of $y$, a soft nearest-neighbour model; as $\alpha \to \infty$ it becomes $k$-nearest neighbours with $k = 1$ |
| $0$ | $0 < \rho < 1$ | ridge regression with the bias left unpenalised |

Compared with plain nearest-neighbour averaging, the local linear fit
($\rho = 0$) is usually better at the edges of the data and wherever the target
has a slope, because a weighted average can only pull predictions towards the
neighbours' values while a local line can follow the trend a little.

## How the case $\rho = 1$ is handled

Yes, $\rho = 1$ is treated separately in the code. The mapping
$\lambda = \rho / (1 - \rho)$ would divide by zero there, and $\lambda = \infty$
cannot be put into a matrix. So when $\rho = 1$ the model skips the linear
solve entirely and returns the normalised weighted average directly:

$$
\hat y(x) = \sum_{i=1}^{n} w_i(x)\, y_i ,
\qquad
\sum_{i=1}^{n} w_i(x) = 1 .
$$

This is exactly the limit of the combined formula as $\lambda \to \infty$, so
there is no jump: predictions for $\rho$ close to $1$ are close to the
predictions for $\rho = 1$. For every $\rho < 1$ the combined formula is used
as written.

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
features are already on comparable scales. The same $\alpha$ then means the
same thing for every column.

**Weights are normalised.** For each query point the weights are divided by
their sum. This matters because the ridge term is not scale-free: multiplying
all weights by a constant $c$ is the same as dividing $\lambda$ by $c$.
Normalising keeps the meaning of $\rho$ the same at every query point and for
every $\alpha$. Before normalising, the smallest distance is subtracted from all
distances, $w_i = \exp(-\alpha (\lVert x_i - x\rVert - \min_j \lVert x_j - x\rVert))$,
so that a very large $\alpha$ cannot underflow every weight to zero. The
normalisation removes this constant factor again.

**The bias is never penalised.** $D$ has a zero in the bias position. If the
bias were penalised too, the $\rho = 1$ limit would predict zero instead of the
weighted average.

**Singular systems are handled.** With $\rho = 0$ and a large $\alpha$, so few
points carry weight that $A^\top W A$ can be singular. The model solves with
`numpy.linalg.lstsq`, which returns the minimum-norm solution instead of
failing. A small positive $\rho$ is a cleaner fix if you see noisy predictions
in that regime.

**Cost.** There is nothing to learn in `fit`; it only stores the data. Each
call to `predict` computes distances to all $n$ training points and solves one
$(d+1) \times (d+1)$ linear system per query point, so prediction costs about
$O(m\, n\, d)$ for $m$ query points. This is fine for thousands of training
points; for very large training sets a neighbour index would be needed to cut
the distance computations.

## Files

- `lwlr.py` — the `LocallyWeightedRegressor` class.
- `README.md` — this file.
