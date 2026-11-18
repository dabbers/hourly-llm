"""Queue-side affinity headroom with a time limit instead of an overtake count (first-order replay of finished runs).
  age T:    nobody is overtaken once the head of the queue has waited T ms
  budget B: a waiter can be overtaken only within B ms of the first time it was overtaken"""
import glob
import sqlite3
import sys

import numpy as np
import pandas as pd

TTL, EPS, LOOK = 120.0, 0.150, 8
RULES = [("count", 3), ("age", 500), ("age", 1000), ("age", 2000), ("budget", 250), ("budget", 500), ("budget", 1000), ("none", 0)]

for d in sorted(glob.glob(sys.argv[1])):
    c = sqlite3.connect(f"{d}/meter.sqlite")
    r = pd.read_sql("select * from requests where t_admitted is not null and replica is not null", c)
    r = r.sort_values("t_admitted").reset_index(drop=True)
    home, last = [], {}
    for conv, rep, t in zip(r.conversation_id, r.replica, r.t_admitted):
        e = last.get(conv) if conv is not None else None
        home.append(e[0] if e and t - e[1] <= TTL else None)
        if conv is not None:
            last[conv] = (rep, t)
    r["home"] = home
    free = r[r.reservation_id.isna() & (r.lazy == 0)].reset_index(drop=True)
    q = free[((free.t_admitted - free.t_arrival) > EPS) & free.affinity_wait_ms.isna()].reset_index(drop=True)
    arr, adm, hm, rep, shp = q.t_arrival.values, q.t_admitted.values, q.home.values, q.replica.values, q["shape"].values
    order = np.argsort(arr)
    n = len(q)
    miss = ((q.home != q.replica) & q.home.notna()).sum()
    turns = free.home.notna().sum()
    miss_tok_share = q[(q.home != q.replica) & q.home.notna()].prompt_tokens.sum() / r.prompt_tokens.sum()
    print(f"\n{d.replace(chr(92), '/').split('/')[-1]}: queue misses {miss}, their prompt tokens {miss_tok_share:.1%} of all")
    waits = [[j for j in order if arr[j] < adm[i] and adm[j] > adm[i]][: LOOK - 1] for i in range(n)]
    for kind, v in RULES:
        used, conv_all, conv_same = set(), 0, 0
        count = np.zeros(n, int)
        first = np.full(n, np.nan)
        for i in range(n):
            t = adm[i]
            if hm[i] == rep[i]:
                continue
            win = waits[i]
            if kind == "age" and (t - arr[i]) * 1000 > v:
                continue
            pick = None
            for pos, j in enumerate(win):
                if hm[j] != rep[i] or rep[j] == rep[i] or j in used:
                    continue
                passed = [i, *win[:pos]]
                if kind == "count" and any(count[x] >= v for x in passed):
                    continue
                if kind == "budget" and any(not np.isnan(first[x]) and (t - first[x]) * 1000 > v for x in passed):
                    continue
                pick = (j, passed)
                break
            if pick:
                j, passed = pick
                used.add(j)
                conv_all += 1
                conv_same += shp[j] == shp[i]
                for x in passed:
                    count[x] += 1
                    if np.isnan(first[x]):
                        first[x] = t
        # extra delay proxy: time from a waiter's first overtake to its real admission (upper bound on what the rule adds)
        ov = ~np.isnan(first)
        span = (adm[ov] - first[ov]) * 1000 if ov.any() else np.array([0.0])
        cs_lo, cs_hi = (x / max(1, miss) * miss_tok_share * 76 for x in (conv_same, conv_all))
        print(f"  {kind:>6} {v:5d}: converted {conv_same:4d}-{conv_all:4d} ({conv_same/max(1,miss):.0%}-{conv_all/max(1,miss):.0%} of queue misses), "
              f"hit rate +{conv_same/turns*100:.1f} to +{conv_all/turns*100:.1f} pts, cached share about +{cs_lo:.1f} to +{cs_hi:.1f} pts; "
              f"overtaken {int(ov.sum())} requests, max {count.max()} times")
