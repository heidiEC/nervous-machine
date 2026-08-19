#!/usr/bin/env python3
"""
DOMAIN 2 — hydrology / river streamflow (real USGS NWIS daily discharge).

Predict downstream discharge (Potomac at Washington DC, 01646500) from two upstream
gauges (Potomac at Point of Rocks 01638500, Monocacy at Jug Bridge 01643000), via
nm_core. Honest baselines:
  - TIME-ordered forward holdout vs PERSISTENCE (beat last-value carry-forward = real skill)
  - SHUFFLED within-distribution vs MEAN (is there an instantaneous relationship at all?)
"""
import json, ssl, math, random, urllib.request
import nm_core
random.seed(0)
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

UP = {"01638500": "PointOfRocks", "01643000": "Monocacy"}
TGT = "01646500"
sites = list(UP) + [TGT]
url = ("https://waterservices.usgs.gov/nwis/dv/?format=json&sites=" + ",".join(sites)
       + "&parameterCd=00060&startDT=2014-01-01&endDT=2023-12-31")
print("fetching USGS daily discharge 2014-2023 ...")
raw = json.load(urllib.request.urlopen(url, context=ctx, timeout=90))
series = {}
for ts in raw["value"]["timeSeries"]:
    site = ts["sourceInfo"]["siteCode"][0]["value"]
    for v in ts["values"][0]["value"]:
        try:
            series.setdefault(site, {})[v["dateTime"][:10]] = float(v["value"])
        except (ValueError, TypeError):
            pass
for s in sites:
    print(f"  {s} {UP.get(s, 'TARGET'):14s}: {len(series.get(s, {}))} daily values")

DRV = list(UP)
dates = sorted(set(series.get(TGT, {})) & set.intersection(*[set(series.get(s, {})) for s in DRV]))
rows = []
for d in dates:
    if series[TGT].get(d, 0) > 0 and all(series[s].get(d, 0) > 0 for s in DRV):
        rows.append(([math.log(series[s][d]) for s in DRV], math.log(series[TGT][d])))
print(f"{len(rows)} aligned daily rows\n")


def zstats(xs):
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    return mu, sd


def evaluate(rows, names, order, baseline, label):
    n = len(rows); ntr = int(n * 0.7); idx = order
    ds = [zstats([rows[i][0][j] for i in idx[:ntr]]) for j in range(len(names))]
    ymu, ysd = zstats([rows[i][1] for i in idx[:ntr]])
    zd = lambda v: [(v[j] - ds[j][0]) / ds[j][1] for j in range(len(v))]
    zy = lambda v: (v - ymu) / ysd
    edges = [nm_core.Edge(nm, "y") for nm in names]
    for i in idx[:ntr]:
        Z = zd(rows[i][0]); yy = zy(rows[i][1])
        for j, e in enumerate(edges):
            e.learn(Z[j], yy)
    em, eb = [], []
    for i in idx[ntr:]:
        yy = zy(rows[i][1]); Z = zd(rows[i][0])
        num = den = 0.0
        for j, e in enumerate(edges):
            num += e.Z * e.predict(Z[j]); den += e.Z
        pred = num / den if den > 0 else 0.0
        base = zy(rows[i - 1][1]) if baseline == "persistence" else 0.0
        em.append((pred - yy) ** 2); eb.append((base - yy) ** 2)
    rm = (sum(em) / len(em)) ** 0.5; rb = (sum(eb) / len(eb)) ** 0.5
    print(f"[{label}]")
    print(f"  SKILL = {(1 - rm / rb) * 100:+.1f}%   (model RMSE {rm:.3f} vs {baseline} {rb:.3f})")
    for j, nm in enumerate(names):
        e = edges[j]
        print(f"    {UP[nm]:14s} W={e.W:+.3f}  Z={e.Z:.3f}")


evaluate(rows, DRV, list(range(len(rows))), "persistence", "TIME-ordered forward vs PERSISTENCE")
print()
sh = list(range(len(rows))); random.shuffle(sh)
evaluate(rows, DRV, sh, "mean", "SHUFFLED within-distribution vs MEAN")
