// Jev via OpenRouter's Decisions API (https://openrouter.ai/api/alpha/decisions).
// Same state+questions shape as TypeSafe's endpoint, so results are comparable
// when we later switch MODEL=jev on the direct TypeSafe key.
import type { Action, Decision, Model, TradeState } from "./model";
import { QUESTIONS } from "./model";

export class OpenRouterJevModel implements Model {
  readonly name = "jev-openrouter";
  private apiKey = process.env.OPENROUTER_API_KEY ?? "";
  private modelId = process.env.OPENROUTER_JEV_MODEL ?? "typesafe/jev-1.13";
  private url = process.env.OPENROUTER_DECISIONS_URL ?? "https://openrouter.ai/api/alpha/decisions";

  async decide(state: TradeState): Promise<Decision> {
    if (!this.apiKey) throw new Error("OPENROUTER_API_KEY is not set");
    const t0 = performance.now();
    const res = await fetch(this.url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model: this.modelId, state, questions: QUESTIONS }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`OpenRouter decisions ${res.status}: ${text.slice(0, 300)}`);
    }
    const r = await res.json();
    const a = r?.answers?.direction;
    if (!a || a.type !== "choice") {
      throw new Error(`unexpected decisions response: ${JSON.stringify(r).slice(0, 300)}`);
    }
    const p = a.probabilities ?? {};
    const buy = p.buy ?? 0, sell = p.sell ?? 0;
    return {
      action: (a.choice === "sell" ? "sell" : "buy") as Action,
      probabilities: { buy, sell, hold: 0 },
      upIn10: buy,
      latencyMs: performance.now() - t0,
      inputTokens: r?.usage?.input_tokens ?? 0,
    };
  }
}