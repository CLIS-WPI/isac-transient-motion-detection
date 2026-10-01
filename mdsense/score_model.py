"""Per-observation-pattern LLR: Gaussian class-conditionals on the 2-D window features,
fitted on the train split only. The same model family is used for every policy."""
from dataclasses import dataclass
import numpy as np


class GaussianLLR:
    def __init__(self, clip=8.0, ridge=1e-2):
        self.clip, self.ridge = clip, ridge

    def fit(self, X0, X1):
        X0, X1 = np.asarray(X0, float), np.asarray(X1, float)
        if X0.ndim == 1:
            X0 = X0[:, None]
        if X1.ndim == 1:
            X1 = X1[:, None]
        if len(X0) < 10 or len(X1) < 10:
            raise ValueError(f"too few windows to fit LLR (null={len(X0)}, event={len(X1)})")
        self.params = []
        for X in (X0, X1):
            mu = X.mean(0)
            if X.shape[1] == 1:
                S = np.array([[float(np.var(X[:, 0], ddof=1) + self.ridge)]])
            else:
                S = np.cov(X.T) + self.ridge * np.eye(X.shape[1])
            self.params.append((mu, np.linalg.inv(S), np.linalg.slogdet(S)[1]))
        self.n0, self.n1 = len(X0), len(X1)
        return self

    def _logpdf(self, x, k):
        mu, P, ld = self.params[k]
        d = np.asarray(x, float).reshape(-1) - mu
        return -0.5 * (d @ P @ d + ld)

    def __call__(self, x):
        return float(np.clip(self._logpdf(x, 1) - self._logpdf(x, 0), -self.clip, self.clip))


@dataclass
class SliceLLR:
    """Picklable view of a fitted LLR on a subset of features (energy and/or Doppler)."""
    base: GaussianLLR
    index: tuple

    def __call__(self, x):
        return self.base(np.asarray(x, float).reshape(-1)[list(self.index)])
