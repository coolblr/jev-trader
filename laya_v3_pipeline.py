"""
laya_v3_pipeline.py -- fine-tuned regression head, ModernBERT-large (head-only),
predicting FORWARD 100-block PATH VOLATILITY (realized variance) from the
exact TradeState Jev saw (states.json), joined against realized mids
(events-run3.json). Same inputs as v2's direction classifier -- new target.

Why this experiment, not another direction rerun:

vol_step0_reanalysis.py and vol_baseline_clustering.py (run first, $0 cost,
reusing already-collected data) established:
  - Jev's own direction-confidence carries no information about forward
    volatility, net-displacement or path-based (both FAIL |rho|>0.20).
  - Trailing 100-block realized volatility DOES predict forward 100-block
    realized volatility on this dataset: rho=+0.354 on ~independent windows
    (n=471, p=2.5e-15). This is real volatility clustering, not noise.

That reframes the bar. Beating zero is not interesting -- trailing-RV
already clears the old kind of bar. The only interesting question left is
whether a model with access to the FULL order-book state (depth bands,
book imbalance, CVD flow -- not just trailing returns) can beat trailing-RV
itself out of sample.

Pre-registered bar (set BEFORE running): test Spearman rho between the
model's output and actual forward RV must exceed the trailing-RV
baseline's rho on the SAME test rows by >= +0.05 absolute. A model that
merely matches trailing-RV isn't worth the fine-tuning cost -- the trailing
feature is already sitting in the state text for free (returnsBps).

Split is time-ordered AND purged: any row whose forward window
[block, block+100] would cross a train/val/test boundary is dropped from
the earlier split, so no label leaks return data into the next split's
training/eval region.

Usage (from the folder containing states.json and events-run3.json):
  py laya_v3_pipeline.py --states states.json --events events-run3.json --out laya-ft-v3
"""
import argparse
import random

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, set_seed)

from laya_v2_pipeline import parse_states, parse_events_mids, state_to_text

HORIZON = 100
RHO_MARGIN_BAR = 0.05


def realized_variance(log_ret, start, length):
    seg = log_ret[start:start + length]
    if len(seg) < length:
        return None
    return float(np.sum(seg ** 2))


def build_rows(states, blocks, mids):
    log_ret = np.diff(np.log(mids))  # log_ret[k] = return blocks[k] -> blocks[k+1]
    n = len(blocks)
    rows = []
    for b, st in states.items():
        i = int(np.searchsorted(blocks, b))
        if i >= n or blocks[i] != b:
            # fall back to nearest block at/after b, same tolerance v2 used for lookahead
            i = int(np.searchsorted(blocks, b))
            if i >= n:
                continue
        if i < HORIZON or i + HORIZON >= n:
            continue
        fwd_rv = realized_variance(log_ret, i, HORIZON)
        trail_rv = realized_variance(log_ret, i - HORIZON, HORIZON)
        if fwd_rv is None or trail_rv is None:
            continue
        rows.append({"block": b, "text": state_to_text(st), "fwd_rv": fwd_rv, "trail_rv": trail_rv})
    rows.sort(key=lambda r: r["block"])
    return rows


def purge_split(rows):
    n = len(rows)
    i1, i2 = int(n * 0.60), int(n * 0.75)
    blk1, blk2 = rows[i1]["block"], rows[i2]["block"]
    train = [r for r in rows[:i1] if r["block"] + HORIZON <= blk1]
    val = [r for r in rows[i1:i2] if r["block"] + HORIZON <= blk2]
    test = rows[i2:]
    dropped = n - len(train) - len(val) - len(test)
    print(f"purged split: train={len(train)} val={len(val)} test={len(test)}  (dropped {dropped} boundary-crossing rows)")
    return train, val, test


class RegDataset(Dataset):
    def __init__(self, rows, tok, target_mean, target_std, max_len=512):
        self.enc = tok([r["text"] for r in rows], truncation=True, max_length=max_len, padding=True)
        log_fwd = np.log(np.array([r["fwd_rv"] for r in rows]) + 1e-12)
        self.targets = (log_fwd - target_mean) / target_std
        self.fwd_rv = np.array([r["fwd_rv"] for r in rows])
        self.trail_rv = np.array([r["trail_rv"] for r in rows])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="laya-ft-v3")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    set_seed(42); random.seed(42); np.random.seed(42)
    states = parse_states(args.states)
    blocks, mids = parse_events_mids(args.events)
    rows = build_rows(states, blocks, mids)
    print(f"states={len(states)} events_blocks={len(blocks)} labeled_rows={len(rows)}")
    if len(rows) < 2000:
        raise SystemExit("too few labeled rows -- check that states and events overlap in block ranges")

    train, val, test = purge_split(rows)

    log_fwd_train = np.log(np.array([r["fwd_rv"] for r in train]) + 1e-12)
    target_mean, target_std = float(log_fwd_train.mean()), float(log_fwd_train.std())
    print(f"train target stats (log fwd RV): mean={target_mean:.4f} std={target_std:.4f}")

    name = "answerdotai/ModernBERT-large"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, problem_type="regression")
    for p in model.model.parameters():
        p.requires_grad = False
    print("head-only: backbone frozen")
    model.cuda()

    train_ds = RegDataset(train, tok, target_mean, target_std)
    val_ds = RegDataset(val, tok, target_mean, target_std)
    test_ds = RegDataset(test, tok, target_mean, target_std)

    def spearman_metric(m):
        preds = m.predictions.reshape(-1)
        labels = m.label_ids.reshape(-1)
        rho, _ = stats.spearmanr(preds, labels)
        return {"spearman": float(rho)}

    common = dict(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=24, per_device_eval_batch_size=48,
        learning_rate=5e-4, logging_steps=25, eval_strategy="epoch",
        save_strategy="no", fp16=True, seed=42, report_to=[])
    try:
        targs = TrainingArguments(warmup_ratio=0.1, weight_decay=0.01, **common)
    except TypeError:
        targs = TrainingArguments(**common)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                       eval_dataset=val_ds, compute_metrics=spearman_metric)
    trainer.train()

    tp = trainer.predict(test_ds)
    model_preds = tp.predictions.reshape(-1)
    fwd_rv_test = test_ds.fwd_rv
    trail_rv_test = test_ds.trail_rv

    model_rho, model_p = stats.spearmanr(model_preds, fwd_rv_test)
    baseline_rho, baseline_p = stats.spearmanr(trail_rv_test, fwd_rv_test)

    print("\n===== TEST (last 25%, purged) =====")
    print(f"n={len(fwd_rv_test)}")
    print(f"model (full-state regression head) vs forward RV:  rho={model_rho:+.4f}  (p={model_p:.2e})")
    print(f"baseline (trailing RV, same test rows) vs forward RV: rho={baseline_rho:+.4f}  (p={baseline_p:.2e})")
    print(f"reference (vol_baseline_clustering.py, thinned, different session slice): rho=+0.354")
    margin = model_rho - baseline_rho
    print(f"\nmargin (model - baseline) = {margin:+.4f}")
    print(f"BAR -> margin >= +{RHO_MARGIN_BAR}: {'PASS' if margin >= RHO_MARGIN_BAR else 'FAIL'}")

    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
