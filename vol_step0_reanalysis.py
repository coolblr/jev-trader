"""
vol_step0_reanalysis.py -- free re-analysis of Stage 1's ALREADY-COLLECTED
Jev decisions (events-run2.json, $5.17 spent Sep 20) to test whether Jev's
own confidence carries information about REALIZED VOLATILITY (|move| bps)
over the same 100-block horizon, even though EXPERIMENT.md Stage 1 showed
zero information about DIRECTION.

This is step 0 of the volatility follow-up described in EXPERIMENT.md's
"Future work" (#12): the cheapest possible test of whether it's worth
funding a real Score/Noul-ladder Jev pass or a fine-tuned regression head.
Zero additional API spend -- reuses data already paid for.

Pre-registered bar (set BEFORE running this script):
  full-sample Spearman rho between confidence and |move_bps| must satisfy
  |rho| > 0.20 to count as a promising signal worth a funded follow-up.
  Below that, stop here -- the qualitative stasis-detector effect described
  in EXPERIMENT.md Stage 1 is real but too weak on its own to build on.

Confidence is defined as |P(buy) - 0.5| * 2, i.e. distance from maximum
uncertainty, rescaled to [0, 1]. Only non-late decisions are scored, same
filter EXPERIMENT.md Stage 1 applied (late decisions carry no real
probabilities -- hold/late always reports upIn10=0.5).

Because horizon windows overlap (100-block horizon, ~1-block decision
stride), adjacent rows are highly non-independent -- more so for a
volatility target than direction, since realized vol clusters in time.
The full-sample correlation is therefore reported alongside a thinned
(non-overlapping, stride=100) version as a robustness check: if the two
disagree sharply, the full-sample number is likely inflated by
autocorrelation, not by a real relationship.

Usage:
  py vol_step0_reanalysis.py --events events-run2.json
"""
import argparse
import json

import numpy as np
from scipy import stats

HORIZON_BLOCKS = 100
RHO_BAR = 0.20


def open_text(path):
    """utf-8 (Bun-written full file) or utf-16LE (PowerShell > slice)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse_events(path):
    """SSE stream -> list of event dicts, one per 'data: {...}' line."""
    records = []
    for line in open_text(path).splitlines():
        if not line.startswith("data: "):
            continue
        try:
            o = json.loads(line[6:])
        except Exception:
            continue
        if isinstance(o, dict) and "block" in o and "mid" in o:
            records.append(o)
    return records


def dedupe_by_block(records):
    """Keep the latest-ts event per block (a block can appear as snapshot +
    block, or be re-emitted); mirrors laya_v2_pipeline.py's join logic."""
    by_block = {}
    for o in records:
        b = o["block"]
        if b not in by_block or o.get("ts", 0) >= by_block[b].get("ts", 0):
            by_block[b] = o
    return by_block


def build_rows(by_block):
    blocks_sorted = np.array(sorted(by_block))
    mids_sorted = np.array([by_block[b]["mid"] for b in blocks_sorted])
    n = len(blocks_sorted)

    rows = []
    for idx, b in enumerate(blocks_sorted):
        o = by_block[b]
        dec = o.get("decision")
        if not dec or dec.get("late"):
            continue
        probs = dec.get("probabilities") or {}
        p_buy = probs.get("buy")
        if p_buy is None:
            continue
        j = int(np.searchsorted(blocks_sorted, b + HORIZON_BLOCKS))
        if j >= n:
            continue
        mid0 = o["mid"]
        move_bps = (mids_sorted[j] - mid0) / mid0 * 10000
        conf = abs(p_buy - 0.5) * 2  # rescale distance-from-0.5 to [0, 1]
        rows.append((b, idx, conf, move_bps))
    rows.sort(key=lambda r: r[0])
    return rows


def spearman_report(conf, abs_move, label):
    rho, p = stats.spearmanr(conf, abs_move)
    pearson_r, pearson_p = stats.pearsonr(conf, abs_move)
    print(f"\n[{label}]  n={len(conf)}")
    print(f"  Spearman rho = {rho:+.4f}  (p={p:.2e})")
    print(f"  Pearson  r   = {pearson_r:+.4f}  (p={pearson_p:.2e})")
    return rho, p


def decile_table(conf, abs_move):
    order = np.argsort(conf)
    conf_s, move_s = conf[order], abs_move[order]
    n = len(conf_s)
    print("\n  confidence decile -> mean |move_bps|  (monotonic decrease = stasis-detector confirmed)")
    for d in range(10):
        lo, hi = int(n * d / 10), int(n * (d + 1) / 10)
        if hi <= lo:
            continue
        print(f"    decile {d}  conf in [{conf_s[lo]:.3f}, {conf_s[hi-1]:.3f}]  "
              f"mean|move_bps|={move_s[lo:hi].mean():.4f}  n={hi-lo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    args = ap.parse_args()

    records = parse_events(args.events)
    by_block = dedupe_by_block(records)
    rows = build_rows(by_block)
    if len(rows) < 500:
        raise SystemExit(f"too few scoreable rows ({len(rows)}) -- check --events path / late-decision filter")

    blocks = np.array([r[0] for r in rows])
    conf = np.array([r[2] for r in rows])
    move_bps = np.array([r[3] for r in rows])
    abs_move = np.abs(move_bps)

    print(f"Parsed {len(records)} raw events -> {len(by_block)} unique blocks -> {len(rows)} scoreable (non-late) decisions")
    print(f"Session span: block {blocks.min()} to {blocks.max()}  |  base |move_bps| mean={abs_move.mean():.4f} median={np.median(abs_move):.4f}")

    rho_full, p_full = spearman_report(conf, abs_move, "FULL SAMPLE (overlapping horizons -- point estimate)")
    decile_table(conf, abs_move)

    # Robustness: thin to ~non-overlapping windows (stride = horizon) so rows
    # are approximately independent. This is the number to trust for "is
    # this real," not the full-sample one, since overlapping 100-block
    # horizons on autocorrelated volatility can inflate rho artificially.
    thin_idx = []
    last_block = -HORIZON_BLOCKS
    for i, b in enumerate(blocks):
        if b - last_block >= HORIZON_BLOCKS:
            thin_idx.append(i)
            last_block = b
    conf_thin, move_thin = conf[thin_idx], abs_move[thin_idx]
    if len(conf_thin) >= 30:
        rho_thin, p_thin = spearman_report(conf_thin, move_thin, "THINNED (stride>=100 blocks -- ~independent rows)")
    else:
        rho_thin, p_thin = float("nan"), float("nan")
        print(f"\n[THINNED] too few independent windows ({len(conf_thin)}) to report -- session likely shorter than expected")

    # Chronological stability: split in half, same direction both halves?
    mid_i = len(rows) // 2
    rho_h1, _ = stats.spearmanr(conf[:mid_i], abs_move[:mid_i])
    rho_h2, _ = stats.spearmanr(conf[mid_i:], abs_move[mid_i:])
    print(f"\n[STABILITY] first-half rho={rho_h1:+.4f}  second-half rho={rho_h2:+.4f}  "
          f"({'same sign' if rho_h1 * rho_h2 > 0 else 'SIGN FLIPS -- treat full-sample rho with suspicion'})")

    print(f"\n===== PRE-REGISTERED BAR: |rho| > {RHO_BAR} =====")
    print(f"  full-sample:  |rho|={abs(rho_full):.4f}  -> {'PASS' if abs(rho_full) > RHO_BAR else 'FAIL'}")
    if not np.isnan(rho_thin):
        print(f"  thinned:      |rho|={abs(rho_thin):.4f}  -> {'PASS' if abs(rho_thin) > RHO_BAR else 'FAIL'}")
        print("\n  Trust the thinned number over the full-sample one for the go/no-go call.")


if __name__ == "__main__":
    main()
