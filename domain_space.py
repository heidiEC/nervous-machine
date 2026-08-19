#!/usr/bin/env python3
"""
DOMAIN 1 — space / thermospheric density.  Real GRACE-FO neutral density vs real OMNI
space-weather drivers, learned with nm_core (the one source of truth).

Two splits, to separate "can't learn it" from "can't generalize across the solar cycle":
  - TIME-ordered forward holdout (train ~2018 solar min, test ~2024+ solar max): the honest hard test.
  - SHUFFLED within-distribution split: can it fit the relationship at all?
Reports held-out skill vs the mean-baseline + the learned per-driver edges.
"""
import json, math, random
from collections import defaultdict
import nm_core
random.seed(0)

OBS = "/Users/heidi.bennett/space-waze/results/learn-gracefo-obs-multiyear.jsonl"
OMNI = "/Users/heidi.bennett/space-waze/results/omni-drivers-2018-2025.jsonl"
DRIVERS = ["dst", "ae_index", "imf_bz", "solar_wind_speed", "solar_wind_density"]

omni = {}
for line in open(OMNI):
    d = json.loads(line); omni[d["timestamp"]] = d

by_vox = defaultdict(list)
for line in open(OBS):
    r = json.loads(line)
    if r.get("o") and r["o"] > 0:
        by_vox[r["v"]].append((r["t"], r["o"]))
vox = max(by_vox, key=lambda k: len(by_vox[k]))
rows = []
for t, o in sorted(by_vox[vox]):
    od = omni.get(t)
    if od and all(od.get(k) is not None for k in DRIVERS):
        rows.append(([float(od[k]) for k in DRIVERS], math.log(o)))


def zstats(xs):
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    return mu, sd


def run(rows, label):
    n = len(rows); ntr = int(n * 0.7)
    ds = [zstats([r[0][i] for r in rows[:ntr]]) for i in range(len(DRIVERS))]
    ymu, ysd = zstats([r[1] for r in rows[:ntr]])
    zd = lambda v: [(v[i] - ds[i][0]) / ds[i][1] for i in range(len(v))]
    zy = lambda v: (v - ymu) / ysd
    edges = {k: nm_core.Edge(k, "y") for k in DRIVERS}
    for vec, y in rows[:ntr]:
        Z, yy = zd(vec), zy(y)
        for i, k in enumerate(DRIVERS):
            edges[k].learn(Z[i], yy)

    def predict(vec):
        Z = zd(vec); num = den = 0.0
        for i, k in enumerate(DRIVERS):
            e = edges[k]; num += e.Z * e.predict(Z[i]); den += e.Z
        return num / den if den > 0 else 0.0
    em, eb = [], []
    for vec, y in rows[ntr:]:
        yy = zy(y); p = predict(vec)
        em.append((p - yy) ** 2); eb.append(yy ** 2)
    rm = (sum(em) / len(em)) ** 0.5; rb = (sum(eb) / len(eb)) ** 0.5
    print(f"\n[{label}]")
    print(f"  SKILL = {(1 - rm / rb) * 100:+.1f}%   (model RMSE {rm:.3f} vs mean-baseline {rb:.3f})")
    for k in DRIVERS:
        e = edges[k]
        print(f"    {k:22s} W={e.W:+.3f}  Z={e.Z:.3f}")


print(f"voxel {vox} | {len(rows)} aligned hourly obs")
run(rows, "TIME-ordered forward holdout (2018 min -> 2024+ max : regime shift)")
sr = rows[:]; random.shuffle(sr)
run(sr, "SHUFFLED within-distribution split (can it fit at all?)")
