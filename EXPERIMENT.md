# Can a Cheap Decision Model Find Edge in a DEX Order Book?
### A falsification experiment with Jev, Laya, and 117,000 Monad blocks
**Venky Labs · September 19–21, 2026 · Total cost: $9.54 in API credits + one weekend**

---

## TL;DR

We tested whether "System One" decision models - Jev (TypeSafe AI, cloud API) and a
locally fine-tuned ModernBERT-large/Laya-style head - can predict 100-block (~30 s)
price direction on the MON/USDC order book (Kuru DEX, Monad mainnet), well enough to
trade profitably as a high-frequency market maker.

**They cannot. Three independent approaches, one consistent negative result:**

| Experiment | Model / input | Test accuracy | Test Brier | Base-rate Brier | Pre-registered bars |
|---|---|---|---|---|---|
| Stage 1 | Jev-1.13 via OpenRouter, full book state, zero-shot | 49.9% (base 52.9%) | 0.408 | 0.249 | FAIL |
| v1 | ModernBERT-large head-only, reconstructed summary features | 52.1% (base 53.8%) | 0.2495 | 0.2486 | FAIL |
| v2 | ModernBERT-large head-only, **full true TradeState, 53K examples** | 50.9% (base 45.9%) | 0.2548 | 0.2483 | FAIL |

The v2 model is *worse than outputting a constant*: with a 45.9% base rate, "always
predict down" scores 54.1% accuracy / Brier 0.2483. Cheap decisions do not create
signal; they only make losing cheaper to discover.

**Two findings are publishable on their own:**
1. Jev's probabilities on order-book state are **flat calibration + extreme overconfidence**
   - a stasis detector, not a direction predictor (details below).
2. The economics: at $0.000067/decision, inference would consume ~40% of gross spread
   capture *even if the model had edge* - decision density, not model quality, is the
   binding constraint for micro-trading agents.

---

## 1. Why this experiment

On 2026-09-15, TypeSafe AI launched Jev, a "System One" decision model claimed to be
20–200x faster and 40–400x cheaper than LLMs, outputting typed decisions (choice /
score / noul) with probabilities. The same week, a Monad engineer published an
open-source trading bot (`jarrodwatts/jev-trader`) that queries such a model every
~300 ms Monad block on the Kuru MON/USDC order book, quoting post-only limit orders
one tick inside the touch.

The implicit claim: cheap, fast judgment could unlock profitable high-frequency
on-chain market making. This project treats that claim as a **falsifiable hypothesis**
and runs the cheapest rigorous experiment that can kill it.

## 2. Hypotheses and pre-registered bars

Set before any data collection:

- **H1 (directional signal):** model probabilities of "higher/lower than mid after
  100 blocks" beat chance. **Bars:** test accuracy > 56% AND test Brier < 0.24
  (time-ordered split; base-rate predictor scores ~0.249 Brier, constant-0.5 scores 0.250).
- **H2 (economics):** a strategy can pay for its own inference. **Bar:** per-fill
  inference cost ≪ gross spread capture (~2–5 bps on a $5 order).
- **H3 (infrastructure):** a consumer Windows box can sustain a 300 ms decision loop.
  (Bar: no manual intervention for multi-hour sessions.)

## 3. Setup

- **Base:** fork of `jarrodwatts/jev-trader` (Bun/TypeScript). Two additions:
  1. `src/model-openrouter.ts` - adapter speaking OpenRouter's Decisions API
     (`POST /api/alpha/decisions`, model `typesafe/jev-1.13`), selected via
     `MODEL=jev-openrouter`; `OPENROUTER_DECISIONS_URL` env override later enables a
     local model with zero further code changes.
  2. Two-line state logger in `src/trader.ts` - every `TradeState` appended to
     `states.json` per block (works with `MODEL=mock`, i.e., **free data collection**).
- **Machine:** Windows 11, RTX 2070 8 GB, Python 3.14, torch 2.11+cu128,
  `laya 0.3.4`, `transformers 5.17`, `peft` (LoRA flag available, unused).
- **Market:** MON/USDC on Kuru (Monad mainnet, ~3.3 blocks/s, 300 ms block time).
- **All runs were dry-run**: real book, real decisions, simulated fills. No wallet,
  no private key, no capital at risk at any point.

## 4. Stage 0 - infrastructure (Sep 19, overnight, mock model)

- ~100K blocks, one decision per block, loop ~115 ms (read 28–46 ms vs public RPC),
  zero trading errors; single benign PowerShell stderr artifact at startup.
- Mock heuristic baseline: −$1.49 sim P&L over ~9.5 h - the adverse-selection tax of
  naive quote-everything market making, measured.
- **H3: PASS.** The 300 ms loop is viable on consumer hardware; this was the
  cheapest big risk to retire first.

## 5. Stage 1 - Jev zero-shot (Sep 20, ~6.8 h, $5.17 inference spend)

79,806 blocks → 63,214 decisions → 4,907 sim fills (6.6% fill rate; 93.5% of quotes
simply replaced next block). Decision latency 140–350 ms via OpenRouter, ~21% of
blocks late (bot correctly held). Bot-reported cost matched OpenRouter billing to the cent.

Scored 14,712 non-late decisions against realized 100-block moves (Sunday dataset,
price +465 bps over the session):

- **Accuracy 49.9%** vs 52.9% base rate. A brick that always said "down" scored 75.2%.
- **Discrimination: E[P(buy) | up] = 0.410 vs E[P(buy) | down] = 0.408** - zero separation.
- **Brier 0.408** vs 0.250 (constant 0.5) and 0.249 (base rate). Confidence
  anti-correlates with accuracy (49.1% high-conf vs 50.8% low-conf).
- **Calibration flat at ~52–53% realized-up across every predicted-probability bin**
  (P = 0.04 → 52.6% up; P = 0.92 → 52.6% up).
- **The stasis-detector effect:** the model is most confident (0.92–0.99) exactly when
  the next 100 blocks move 0.1–0.3 bps, and least confident when they move 2–3 bps.
  It reads book *stillness* as certainty and book *activity* as uncertainty - and
  neither has any bearing on direction.

A 5-minute window from the prior day (frozen book, price falling 75% of horizons)
showed the same pattern in miniature: 59.3% argmax accuracy, below the 75.2%
always-down baseline, Brier 0.326.

Trading P&L: −$0.71 (−0.71% of $100 notional) + $5.17 inference = **−5.9% of
bankroll if live, in one day.**

## 6. v1 - fine-tuning on reconstructed features (Sep 20 evening)

ModernBERT-large, backbone frozen (head-only), fp16 on the RTX 2070. Input: features
reconstructed from the event stream only (mid, spread, 3/10-step momentum, realized
vol, fill intensity) - a *deliberately weaker* stand-in while the true-state logger
collected data in parallel (free, mock model).

- 14,703 labeled rows (events-run2-sample.json), time-ordered 60/15/25 split.
- Training loss pinned at ln 2 ≈ 0.693 by epoch 1 - nothing to fit.
- **Test: acc 0.521, Brier 0.2495 (bars 0.56 / 0.24) → FAIL.**
- Temperature fit hit the grid boundary T = 3.00 with val NLL exactly 0.6931.
- Control: Jev on the same test rows scored 0.508 / 0.4028 - the fine-tuned head is
  a strictly better *probability source* than the cloud API purely by being honest
  about its ignorance, while still carrying no edge.

## 7. v2 - fine-tuning on the full true state (Sep 20–21, overnight)

The real experiment. Input: the exact `TradeState` object Jev saw - three depth bands,
top-5 bid/ask levels, four return lags, CVD flow, VWAP, last trade, rolling mids and
trades - serialized to ~400-token text; ModernBERT-large (8K context) swallows it whole.

- **53,491 states logged → 53,348 labeled examples (99.7% join rate),** this time
  across a session with base rate 48.1% (different regime from Stage 1 - no trend to lean on).
- Time-ordered 32,008 / 8,003 / 13,337.
- `train_loss` 0.639 < ln 2 - the head *learned the training set* (memorizing
  thousands of near-duplicate frozen-book states). `eval_loss` 0.844 - none of it
  transferred. Temperature again pegged at T = 3.00.
- **Test: acc 0.509, Brier 0.2548 vs base-rate 0.2483 → FAIL.**
  The model underperforms predicting a constant.

## 8. Conclusions

1. **H1 decisively rejected.** At the 100-block horizon on this book, direction is not
   predictable from a single snapshot by (a) a frontier structured-decision API
   zero-shot, (b) a fine-tuned encoder on summary features, or (c) a fine-tuned
   encoder on the full state with 53K examples. One market, one weekend, one horizon -
   the null is well-supported *for this decision point*.
2. **H2 rejected on arithmetic.** Inference cost per fill ($0.0011) rivals gross
   spread capture (~$0.002–0.003 per $5 fill) *before* adverse selection, which
   reliably turns sim-P&L positive scenarios negative. A model with genuine edge
   would still need decision throttling (e.g., skip frozen-book blocks) to survive.
3. **H3 confirmed.** Consumer hardware runs a 300 ms trading-decision loop for days.
   The bottleneck was never infrastructure.

## 9. Limitations (stated before anyone else does)

- Single pair, single venue, one weekend (Saturday trend-up +4%, Sunday round-trip,
  Monday-adjacent quiet periods). Regime diversity is the strongest argument for
  re-testing - the dataset pipeline makes it cheap.
- Sim fills are *generous*: any print crossing the quote fills it. Live post-only
  fills are adverse-selected, so every P&L figure here is a best case.
- Head-only fine-tuning only; LoRA (`--lora`) and longer horizons remain untried -
  listed under future work, not quietly discarded.
- The label (strict sign of mid move) ignores magnitude; a model could be right about
  volatility and score poorly here. See §10.

## 10. Reproduction

```powershell
# Stage 0/1: bot (Bun). .env: MODEL=mock|jev-openrouter, DRY_RUN=true, PRIVATE_KEY empty
bun run start
curl.exe -N http://localhost:3000/events >> events.jsonl   # second window

# v1 / v2 pipelines (Python, RTX-class GPU)
py laya_v1_pipeline.py --events events-run2-sample.json
py laya_v2_pipeline.py --states states.json --events events-run3.json

# bars are printed and pre-registered in each script's docstring:
# accuracy > 0.56 AND Brier < 0.24 on the time-ordered test split
```

Fork-specific files: `src/model-openrouter.ts`, the two-line logger in `src/trader.ts`,
`laya_v1_pipeline.py`, `laya_v2_pipeline.py`. Raw data (states.json, events*.jsonl,
~200 MB total) not committed; regenerate per §3.

## 11. What survived (reusable assets)

- A validated 300 ms decision loop with correct late/hold handling and graceful
  degradation under API rate limits.
- A general (state, decision, realized-outcome) dataset methodology: log states with
  a mock model for $0, join outcomes by block number, time-ordered split, temperature-fit.
- A working local fine-tuning stack for typed-decision-style heads on consumer GPUs.
- `OPENROUTER_DECISIONS_URL` shim point: a local model can replace the cloud API with
  zero changes to trading logic - the integration path if a future model passes the bars.

## 12. Future work (pre-registered before trying)

1. **Volatility, not direction** - predict |move| over 10–100 blocks. The
   stasis-detector effect suggests book state carries *size* information even though
   it carries none on sign.
2. **More regimes** - weekday sessions, higher-volatility days; the cost is now zero
   (mock collection) plus one fine-tune run.
3. **Horizon sweep** - 10 blocks (3 s) may behave differently from 100.
4. **LoRA / full fine-tune** of the backbone if any of the above shows life.

---

*Ethics/security note: no private keys ever touched a chat, a screenshot, or the repo;
one API key was exposed in a screenshot early on and was rotated within the hour.
All trading was simulated. Nothing here is financial advice - it is, if anything,
anti-financial-advice.*

*License: fork of MIT-licensed `jarrodwatts/jev-trader`; pipeline scripts herein are
MIT. Jev is a trademark of TypeSafe AI; Laya of Convai Innovations. Models referenced
per their respective model cards.*
