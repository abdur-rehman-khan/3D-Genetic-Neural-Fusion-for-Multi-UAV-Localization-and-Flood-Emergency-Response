"""Small NumPy MLP (input -> hidden(ReLU) -> output) with Adam.

The original MATLAB network is featureInputLayer(3) -> FC(32) -> ReLU -> FC(3) -> regression.
This class reproduces that topology (``hidden=(32,)``) and exposes the weights as a flat vector so
a genetic algorithm can optimise them.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (64, 32), n_out: int = 3,
                 rng: np.random.Generator | None = None):
        self.sizes = (n_in, *hidden, n_out)
        rng = rng or np.random.default_rng(0)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), size=(a, b)))
            self.b.append(np.zeros(b))

    @property
    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def get_flat(self) -> np.ndarray:
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(self.W, self.b)])

    def set_flat(self, v: np.ndarray) -> None:
        i = 0
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            self.W[k] = v[i:i + w.size].reshape(w.shape).copy(); i += w.size
            self.b[k] = v[i:i + b.size].copy(); i += b.size

    def forward(self, x: np.ndarray, cache: bool = False):
        a = np.atleast_2d(x); acts = [a]
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = np.maximum(z, 0.0) if k < len(self.W) - 1 else z
            acts.append(a)
        return (a, acts) if cache else a

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def mse(self, x, y) -> float:
        return float(np.mean((self.forward(x) - y) ** 2))

    def fit(self, x, y, epochs: int = 300, lr: float = 3e-3, batch: int = 128,
            grad_clip: float | None = 1.0, rng: np.random.Generator | None = None) -> list[float]:
        """Mini-batch Adam on MSE. Returns the per-epoch training loss."""
        rng = rng or np.random.default_rng(0)
        x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
        params = [*self.W, *self.b]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        t = 0; hist = []
        L = len(self.W)
        for _ in range(epochs):
            idx = rng.permutation(n); tot = 0.0
            for s in range(0, n, batch):
                bi = idx[s:s + batch]; xb, yb = x[bi], y[bi]
                out, acts = self.forward(xb, cache=True)
                delta = 2.0 * (out - yb) / (len(bi) * yb.shape[1])
                gW, gb = [None] * L, [None] * L
                for k in range(L - 1, -1, -1):
                    gW[k] = acts[k].T @ delta; gb[k] = delta.sum(0)
                    if k > 0:
                        delta = (delta @ self.W[k].T) * (acts[k] > 0)
                grads = [*gW, *gb]
                if grad_clip:
                    norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
                    if norm > grad_clip:
                        grads = [g * grad_clip / norm for g in grads]
                t += 1
                for i, (p, g) in enumerate(zip([*self.W, *self.b], grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
                    p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
                tot += float(np.sum((out - yb) ** 2))
            hist.append(tot / (n * y.shape[1]))
        return hist
