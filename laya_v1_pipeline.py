"""
laya_v1_pipeline.py -- Stage 3 v1: fine-tune ModernBERT-large (head-only) to predict
100-block direction from features reconstructed from jev-trader event logs.

Runs locally on RTX 2070 (fp16, backbone frozen). Time-ordered split, temperature
scaling fitted on validation, head-to-head vs Jev's recorded probabilities.

Usage (from the folder containing events-run2-sample.json):
  py laya_v1_pipeline.py --events events-run2-sample.json --out laya-ft

Pre-registered success bars (set BEFORE training):
  test accuracy > 0.56  AND  test Brier < 0.24
Note: v1 uses reconstructed features, not the true book state. A FAIL does not
prove the book has no signal (v2 with real states is the stronger test); a PASS
is a genuine result.
"""
import os, json, io, argparse, random
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)

def parse_sse(path):
    events = []
    with io.open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("data: "):
                continue
            try:
                obj = json.loads(line[6:])
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("block") is not None:
                events.append(obj)
    return events

def build_rows(events, lookahead=100):
    by_block, fills = {}, []
    for e in events:
        if "decision" in e:
            b = e["block"]
            if b not in by_block or e.get("ts", 0) > by_block[b].get("ts", 0):
                by_block[b] = e
        elif isinstance(e.get("fill"), dict):
            fills.append(e["block"])
    blocks = np.array(sorted(by_block))
    mids = np.array([by_block[b]["mid"] for b in blocks])
    late = np.array([by_block[b]["decision"]["late"] for b in blocks])
    fills = np.array(sorted(fills))
    rows, n = [], len(blocks)
    for i in range(n):
        if late[i] or i < 12:
            continue
        j = int(np.searchsorted(blocks, blocks[i] + lookahead))
        if j >= n:
            continue
        mom3 = (mids[i] / mids[i - 3] - 1) * 1e4
        mom10 = (mids[i] / mids[i - 10] - 1) * 1e4
        rets = np.diff(np.log(mids[i - 10:i + 1]))
        vol = float(rets.std() * 1e4) if len(rets) > 1 else 0.0
        f_rate = float(((fills >= blocks[i] - 50) & (fills <= blocks[i])).sum()) / 50.0
        text = (f"market=MON-USDC mid={mids[i]:.6f} "
                f"spread_bps={by_block[blocks[i]]['spreadBps']:.2f} "
                f"mom_3={mom3:+.2f}bps mom_10={mom10:+.2f}bps vol_10={vol:.2f}bps "
                f"fill_rate_50={f_rate:.3f}")
        rows.append({"block": int(blocks[i]), "text": text,
                     "label": 1 if mids[j] > mids[i] else 0,
                     "jev_pbuy": by_block[blocks[i]]["decision"]["probabilities"]["buy"]})
    return rows

class ListDataset(Dataset):
    def __init__(self, rows, tok, max_len=96):
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
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="laya-ft")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lora", action="store_true", help="escalation: LoRA adapters instead of head-only")
    args = ap.parse_args()

    set_seed(42); random.seed(42); np.random.seed(42)
    rows = build_rows(parse_sse(args.events))
    y_all = np.array([r["label"] for r in rows])
    print(f"dataset rows: {len(rows)}  base rate up: {y_all.mean():.3f}")
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

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=2e-4 if args.lora else 1e-3,
        logging_steps=25,
        eval_strategy="epoch", save_strategy="no", fp16=True, seed=42, report_to=[])
    acc_fn = lambda m: {"acc": float((m.predictions.argmax(-1) == m.label_ids).mean())}
    trainer = Trainer(model=model, args=targs,
                      train_dataset=ListDataset(train, tok),
                      eval_dataset=ListDataset(val, tok),
                      compute_metrics=acc_fn)
    trainer.train()

    # temperature scaling fitted on validation
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

    # test
    tp = trainer.predict(ListDataset(test, tok))
    z = tp.predictions / bestT
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    p_up, y = p[:, 1], tp.label_ids
    acc = float(((p_up > 0.5).astype(int) == y).mean())
    brier = float(((p_up - y) ** 2).mean())
    jev = np.array([r["jev_pbuy"] for r in test])
    jev_acc = float(((jev > 0.5).astype(int) == y).mean())
    jev_brier = float(((jev - y) ** 2).mean())

    print("\n===== TEST (last 25% of session) =====")
    print(f"fine-tuned : acc={acc:.3f}  Brier={brier:.4f}  (n={len(y)}, base={y.mean():.3f})")
    print(f"Jev (same rows): acc={jev_acc:.3f}  Brier={jev_brier:.4f}")
    print(f"BARS -> acc>0.56: {'PASS' if acc > 0.56 else 'FAIL'} | "
          f"Brier<0.24: {'PASS' if brier < 0.24 else 'FAIL'}")

    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved to {args.out}/")

if __name__ == "__main__":
    main()
