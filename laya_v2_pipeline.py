"""
laya_v2_pipeline.py -- Stage 3 v2: fine-tune ModernBERT-large (head-only) on the TRUE
TradeState (depth bands, 5-level book, CVD flow, rolling mids/trades) to predict
100-block direction. Labels joined from the bot's event stream (mid at block+100).

Usage (from the folder containing states.json and events-run3.json):
  py laya_v2_pipeline.py --states states.json --events events-run3.json --out laya-ft-v2

Pre-registered success bars (unchanged from v1, set BEFORE any training):
  test accuracy > 0.56  AND  test Brier < 0.24
Reference: v1 (reconstructed features) scored acc=0.521, Brier=0.2495 -> both bars FAILED.
"""
import os, json, io, argparse, random
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)

def open_auto(path):
    """utf-8 (Bun-written full file) or utf-16LE (PowerShell > slice)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return io.StringIO(raw.decode("utf-16"))
    return io.StringIO(raw.decode("utf-8-sig"))

def parse_states(path):
    states = {}
    for line in open_auto(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict) and "state" in o and "block" in o:
            states[o["block"]] = o["state"]   # dedupe by block, keep first
    return states

def parse_events_mids(path):
    by_block = {}
    for line in open_auto(path):
        line = line.rstrip("\n")
        if not line.startswith("data: "):
            continue
        try:
            o = json.loads(line[6:])
        except Exception:
            continue
        if isinstance(o, dict) and "decision" in o and "block" in o:
            b = o["block"]
            if b not in by_block or o.get("ts", 0) > by_block[b].get("ts", 0):
                by_block[b] = o
    blocks = np.array(sorted(by_block))
    mids = np.array([by_block[b]["mid"] for b in blocks])
    return blocks, mids

def state_to_text(st):
    t = st["trades"]
    vwap = f"{t['vwap']:.6f}" if t.get("vwap") is not None else "none"
    lastp = f"{t['lastPrice']:.6f}" if t.get("lastPrice") is not None else "none"
    lasts = t.get("lastSide") or "none"
    mids_hist = " ".join((st.get("recentMids") or "").split()[-4:])
    rt = (st.get("recentTrades") or [])[-6:]
    parts = [
        f"market {st['market']} mid {st['mid']:.6f} spread_bps {st['spreadBps']:.2f} "
        f"book_imbalance {st['bookImbalance']:.3f}",
        "depth " + " | ".join(f"{k} bid {v['bid']:.0f} ask {v['ask']:.0f}"
                              for k, v in st["depth"].items()),
        "bids " + " ".join(st["book"]["bids"]),
        "asks " + " ".join(st["book"]["asks"]),
        f"returns_bps last1 {st['returnsBps']['last1']:+.2f} last5 {st['returnsBps']['last5']:+.2f} "
        f"last20 {st['returnsBps']['last20']:+.2f} last100 {st['returnsBps']['last100']:+.2f}",
        f"recent_mids {mids_hist}" if mids_hist else "recent_mids none",
        f"trades count {t['count']} buy_mon {t['buyMon']:.0f} sell_mon {t['sellMon']:.0f} "
        f"cvd_mon {t['cvdMon']:+.0f} vwap {vwap} last_price {lastp} last_side {lasts}",
    ]
    if rt:
        parts.append("recent_trades " + " | ".join(rt))
    return " ".join(parts)

def build_rows(states, blocks, mids, lookahead=100):
    rows = []
    n = len(blocks)
    for b, st in states.items():
        j = int(np.searchsorted(blocks, b + lookahead))
        if j >= n:
            continue
        rows.append({"block": b, "text": state_to_text(st),
                     "label": 1 if mids[j] > st["mid"] else 0})
    rows.sort(key=lambda r: r["block"])
    return rows

class ListDataset(Dataset):
    def __init__(self, rows, tok, max_len=512):
        self.enc = tok([r["text"] for r in rows], truncation=True,
                       max_length=max_len, padding=True)
        self.labels = [r["label"] for r in rows]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.labels[i])
        return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="laya-ft-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()

    set_seed(42); random.seed(42); np.random.seed(42)
    states = parse_states(args.states)
    blocks, mids = parse_events_mids(args.events)
    rows = build_rows(states, blocks, mids)
    y_all = np.array([r["label"] for r in rows])
    print(f"states={len(states)} events_blocks={len(blocks)} labeled_rows={len(rows)}  base rate up: {y_all.mean():.3f}")
    if len(rows) < 2000:
        raise SystemExit("too few labeled rows -- check that states and events overlap in block ranges")
    n = len(rows); i1, i2 = int(n * 0.60), int(n * 0.75)
    train, val, test = rows[:i1], rows[i1:i2], rows[i2:]
    print(f"split (time-ordered): train={len(train)} val={len(val)} test={len(test)}")

    name = "answerdotai/ModernBERT-large"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["qkv_proj", "out_proj"]))
        model.print_trainable_parameters()
    else:
        for p in model.model.parameters():
            p.requires_grad = False
        print("head-only: backbone frozen")
    model.cuda()

    common = dict(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=24, per_device_eval_batch_size=48,
        learning_rate=2e-4 if args.lora else 1e-3,
        logging_steps=25, eval_strategy="epoch", save_strategy="no",
        fp16=True, seed=42, report_to=[])
    try:
        targs = TrainingArguments(warmup_ratio=0.1, weight_decay=0.01, **common)
    except TypeError:
        targs = TrainingArguments(**common)
    acc_fn = lambda m: {"acc": float((m.predictions.argmax(-1) == m.label_ids).mean())}
    trainer = Trainer(model=model, args=targs,
                      train_dataset=ListDataset(train, tok),
                      eval_dataset=ListDataset(val, tok),
                      compute_metrics=acc_fn)
    trainer.train()

    vp = trainer.predict(ListDataset(val, tok))
    logits, labels = vp.predictions, vp.label_ids
    bestT, bestNLL = 1.0, float("inf")
    for T in np.arange(0.4, 3.05, 0.05):
        z = logits / T
        lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) + z.max(1, keepdims=True)
        nll = -(z - lse)[np.arange(len(labels)), labels].mean()
        if nll < bestNLL:
            bestNLL, bestT = float(nll), float(T)
    print(f"temperature fit on val: T={bestT:.2f} (val NLL={bestNLL:.4f})")

    tp = trainer.predict(ListDataset(test, tok))
    z = tp.predictions / bestT
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    p_up, y = p[:, 1], tp.label_ids
    acc = float(((p_up > 0.5).astype(int) == y).mean())
    brier = float(((p_up - y) ** 2).mean())
    br = float(y.mean())
    print("\n===== TEST (last 25% of session) =====")
    print(f"v2 fine-tuned: acc={acc:.3f}  Brier={brier:.4f}  (n={len(y)}, base={br:.3f}, base-rate Brier={br*(1-br):.4f})")
    print(f"reference v1 : acc=0.521  Brier=0.2495")
    print(f"BARS -> acc>0.56: {'PASS' if acc > 0.56 else 'FAIL'} | "
          f"Brier<0.24: {'PASS' if brier < 0.24 else 'FAIL'}")
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved to {args.out}/")

if __name__ == "__main__":
    main()
