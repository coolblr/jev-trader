"""
vol_baseline_clustering.py -- second free check, still $0 marginal cost.

vol_step0_reanalysis.py tested whether Jev's own direction-confidence
predicts NET DISPLACEMENT |move_bps| over the next 100 blocks. It failed
the pre-registered bar (|rho|=0.02 full-sample, 0.007 thinned -- not
distinguishable from noise).

This script asks two different, more important questions using the SAME
already-logged mid series (events-run2.json), before funding any new Jev
call:

1. Does TRAILING realized volatility predict FORWARD realized volatility
   on this market? (The "volatility clustering" baseline any model -- Jev
   or fine-tuned -- has to beat to be worth anything. If this baseline is
   already strong, a model needs to add something beyond it, not just beat
   a coin flip.)

2. Does Jev's direction-confidence predict PATH volatility (forward
   realized variance, i.e. how much the price whipped around) rather than
   NET displacement? A round-trip window (+3bps then -3bps) nets to ~0
   displacement but is not calm -- step0's displacement-based test could
   have missed a real relationship for exactly this reason.

Realized variance over a window = sum of squared consecutive log returns
in that window (standard RV estimator). Trailing and forward windows are
both 100 blocks, matching the bot's HORIZON_BLOCKS.

Pre-registered bar (set before running): |rho| > 0.20, same threshold as
step0, for either question to count as promising.

Usage:
  py vol_baseline_clustering.py --events events-run2.json
"""
import argparse

import numpy as np
from scipy import stats

from vol_step0_reanalysis import parse_events, dedupe_by_block

HORIZON_BLOCKS = 100
RHO_BAR = 0.20


def build_series(by_block):
    blocks = np.array(sorted(by_block))
    mids = np.array([by_block[b]["mid"] for b in blocks])
    confs = []
    for b in blocks:
        dec = by_block[b].get("decision")
        if dec and not dec.get("late") and dec.get("probabilities", {}).get("buy") is not None:
            confs.append(abs(dec["probabilities"]["buy"] - 0.5) * 2)
        else:
            confs.append(np.nan)
    return blocks, mids, np.array(confs)


def realized_variance(log_returns, start, length):
    seg = log_returns[start:start + length]
    return float(np.sum(seg ** 2))


def report(x, y, label):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho, p = stats.spearmanr(x, y)
    print(f"\n[{label}]  n={len(x)}")
    print(f"  Spearman rho = {rho:+.4f}  (p={p:.2e})  -> {'PASS' if abs(rho) > RHO_BAR else 'FAIL'} (bar |rho|>{RHO_BAR})")
    return rho, p, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    args = ap.parse_args()

    records = parse_events(args.events)
    by_block = dedupe_by_block(records)
    blocks, mids, confs = build_series(by_block)
    n = len(blocks)
    print(f"Series: {n} blocks, {blocks.min()}..{blocks.max()}, {np.isfinite(confs).sum()} with scoreable confidence")

    log_ret = np.diff(np.log(mids))  # log_ret[i] = return from block i to i+1
    # trailing_RV[i]: variance over the 100 returns ending just before index i
    # forward_RV[i]:  variance over the 100 returns starting at index i
    valid = np.arange(HORIZON_BLOCKS, n - HORIZON_BLOCKS)
    trailing_rv = np.array([realized_variance(log_ret, i - HORIZON_BLOCKS, HORIZON_BLOCKS) for i in valid])
    forward_rv = np.array([realized_variance(log_ret, i, HORIZON_BLOCKS) for i in valid])
    conf_at = confs[valid]

    print("\n===== Q1: does trailing realized vol predict forward realized vol (clustering baseline)? =====")
    rho_full, p_full, _ = report(trailing_rv, forward_rv, "clustering, full-sample (overlapping)")

    # Thin with stride 200 (trailing+forward windows are each 100, so 200
    # apart guarantees no shared return between consecutive sampled windows)
    thin_idx = np.arange(0, len(valid), 2 * HORIZON_BLOCKS)
    if len(thin_idx) >= 30:
        report(trailing_rv[thin_idx], forward_rv[thin_idx], "clustering, thinned (stride=200, ~independent)")
    else:
        print(f"\n[clustering, thinned] too few independent windows ({len(thin_idx)}) in this session to report")

    print("\n===== Q2: does Jev's direction-confidence predict forward PATH volatility (not net displacement)? =====")
    report(conf_at, forward_rv, "confidence vs forward RV, full-sample (overlapping)")
    thin_idx2 = np.arange(0, len(valid), HORIZON_BLOCKS)
    if np.isfinite(conf_at[thin_idx2]).sum() >= 30:
        report(conf_at[thin_idx2], forward_rv[thin_idx2], "confidence vs forward RV, thinned (stride=100)")


if __name__ == "__main__":
    main()
