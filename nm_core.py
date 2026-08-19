#!/usr/bin/env python3
"""
nm_core.py — Nervous Machine learning core. THE single source of truth for the math.

Learning is computed HERE and only here (and at the server that imports this). No
second copy of the update rule should ever exist again — N copies with no source of
truth is exactly what caused 8 months of drift. If you need the math elsewhere, call
the server; do not re-implement it.

The loop, per driver -> observable edge:

    pred = W * d                          prediction = contribution weight x driver value
    eps  = pred - y                       pure prediction error
    eta  = 1 / (1 + exp(10*(Z - 0.5)))    adaptive inertia: agile when ignorant, ~0 when sure
    W   += -base * eta * eps * d          GRADIENT update -- the driver value d is the term
                                          that was MISSING; it carries the direction that
                                          makes the weight actually converge to the driver's
                                          contribution. Without it the rule only stabilizes a
                                          prior; it never learns.
    Z    : logit certainty, scored on SKILL (this edge's error vs just guessing the mean),
           so an edge that predicts nothing can never read confident.

Dependency-free (stdlib only): testable in isolation, portable to any compile target.
"""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field

# --- certainty kernel: unclamped logit + persistence integral ----------------
# Z = sigmoid(theta), so Z asymptotes and NEVER reaches exactly 1.0. The integral I
# is not gated by eta, so a confident edge still loses certainty under *sustained*
# error while transient noise averages out. (Conformance-pinned reference math.)
LAM, KAPPA, K, Z_CENTER = 0.25, 0.15, 10.0, 0.5


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def logit(p: float) -> float:
    eps = 1e-6; p = min(1.0 - eps, max(eps, p))      # clamps the SEED prob only
    return math.log(p / (1.0 - p))


def certainty_step(theta, I, r, r_ref, lam=LAM, kappa=KAPPA, k=K, z_center=Z_CENTER):
    """One unclamped logit + persistence-integral certainty update.
    r in [0,1] is the inaccuracy signal; g = r_ref - r is signed evidence (support>0)."""
    z = sigmoid(theta)
    g = r_ref - r
    I = (1.0 - lam) * I + lam * g
    eta = 1.0 / (1.0 + math.exp(k * (z - z_center)))
    theta = theta + eta * g + kappa * I
    return theta, I


def learning_rate(z: float, k: float = K, z_center: float = Z_CENTER) -> float:
    """eta(Z): high when ignorant, ~0 when certain."""
    return 1.0 / (1.0 + math.exp(k * (z - z_center)))


# --- the edge: a learnable contribution weight + an honest certainty ---------
@dataclass
class Edge:
    source: str
    target: str
    W: float = 0.3                 # contribution weight = the predictor (seed = prior)
    base_rate: float = 0.3         # gradient step scale (stability)
    skill_ref: float = 0.9         # must beat the mean-baseline by >10% to gain Z
    skill_window: int = 20
    min_skill_obs: int = 5
    z0: float = 0.30
    theta: float = None
    I: float = 0.0
    re: deque = field(default=None, repr=False)    # recent residuals
    ry: deque = field(default=None, repr=False)    # recent targets (mean-baseline)
    n_obs: int = 0

    def __post_init__(self):
        if self.theta is None:
            self.theta = logit(self.z0)
        self.re = deque(maxlen=self.skill_window)
        self.ry = deque(maxlen=self.skill_window)

    @property
    def Z(self) -> float:
        return sigmoid(self.theta)

    def predict(self, d: float) -> float:
        return self.W * d

    def learn(self, d: float, y: float) -> dict:
        pred = self.W * d
        eps = pred - y
        eta = learning_rate(self.Z)
        # GRADIENT weight update: the driver value d makes the weight converge to
        # the driver's true contribution (this is the term that was lost to drift).
        self.W = max(-1.0, min(1.0, self.W - self.base_rate * eta * eps * d))
        # SKILL-based certainty: this edge's recent error vs predicting the mean.
        self.re.append(eps); self.ry.append(y); self.n_obs += 1
        if len(self.re) >= self.min_skill_obs:
            erms = (sum(e * e for e in self.re) / len(self.re)) ** 0.5
            brms = (sum(v * v for v in self.ry) / len(self.ry)) ** 0.5 or 1.0
            r = min(1.0, erms / brms)                 # r<1 => beats the mean baseline
            self.theta, self.I = certainty_step(self.theta, self.I, r, self.skill_ref)
        return {"pred": pred, "eps": eps, "eta": eta, "W": self.W, "Z": self.Z}


if __name__ == "__main__":
    # self-proof: a real driver is learned + trusted; a spurious one is rejected.
    import random
    random.seed(1)

    def zs(xs):
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
        return [(x - mu) / sd for x in xs]

    N = 600
    d1 = [random.gauss(0, 1) for _ in range(N)]
    d2 = [random.gauss(0, 1) for _ in range(N)]
    y = [0.7 * d1[i] - 0.4 * d2[i] + random.gauss(0, 0.25) for i in range(N)]
    D1, Y = zs(d1), zs(y)
    junk = zs(d2)[:]; random.shuffle(junk)

    real = Edge("d1", "y")
    spur = Edge("d2_shuffled", "y")
    for i in range(N):
        real.learn(D1[i], Y[i]); spur.learn(junk[i], Y[i])

    print(f"real driver    : W={real.W:+.3f}  Z={real.Z:.3f}   (expect W~+0.83, Z high)")
    print(f"spurious driver: W={spur.W:+.3f}  Z={spur.Z:.3f}   (expect W~0,    Z~0)")
    ok = real.W > 0.6 and real.Z > 0.8 and abs(spur.W) < 0.4 and spur.Z < 0.2
    print("SELF-TEST:", "PASS" if ok else "FAIL")
