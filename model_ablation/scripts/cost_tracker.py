"""
cost_tracker.py -- LLM API cost tracking for model ablation experiments.

Tracks token usage and cost across Fireworks API calls for GLM-5 and DeepSeek-v3.

Pricing (per 1M tokens):
  - glm-5:        $1.00 input / $3.20 output
  - deepseek-v3:  $0.56 input / $1.68 output

Budget: $34.00 USD total.
"""

import json
import os
import time

import numpy as np

np.random.seed(42)


# Pricing per 1M tokens (USD)
PRICING = {
    "glm-5": {
        "input": 1.00,
        "output": 3.20,
    },
    "deepseek-v3": {
        "input": 0.56,
        "output": 1.68,
    },
}

BUDGET_USD = 34.00


class CostTracker:
    """Track API token usage and cost for LLM experiments.

    Usage:
        tracker = CostTracker()
        tracker.add_usage("deepseek-v3", prompt_tokens=500, completion_tokens=10)
        tracker.add_usage("glm-5", prompt_tokens=600, completion_tokens=8)
        tracker.print_summary()
        tracker.check_budget()  # raises if over budget
    """

    def __init__(self, budget=BUDGET_USD):
        self.budget = budget
        self.usage = {}  # model -> {input_tokens, output_tokens, calls, cost_input, cost_output}
        self.start_time = time.time()

    def add_usage(self, model_name, prompt_tokens, completion_tokens):
        """Record token usage for a single API call.

        Parameters
        ----------
        model_name : str
            Short model name: 'glm-5' or 'deepseek-v3'.
        prompt_tokens : int
            Number of input/prompt tokens.
        completion_tokens : int
            Number of output/completion tokens.
        """
        if model_name not in self.usage:
            self.usage[model_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "cost_input": 0.0,
                "cost_output": 0.0,
            }

        pricing = PRICING.get(model_name)
        if pricing is None:
            # Unknown model -- track tokens but cannot compute cost
            self.usage[model_name]["input_tokens"] += prompt_tokens
            self.usage[model_name]["output_tokens"] += completion_tokens
            self.usage[model_name]["calls"] += 1
            return

        cost_in = prompt_tokens * pricing["input"] / 1_000_000
        cost_out = completion_tokens * pricing["output"] / 1_000_000

        entry = self.usage[model_name]
        entry["input_tokens"] += prompt_tokens
        entry["output_tokens"] += completion_tokens
        entry["calls"] += 1
        entry["cost_input"] += cost_in
        entry["cost_output"] += cost_out

    def total_cost(self):
        """Return total cost across all models."""
        total = 0.0
        for entry in self.usage.values():
            total += entry.get("cost_input", 0.0) + entry.get("cost_output", 0.0)
        return total

    def remaining_budget(self):
        """Return remaining budget in USD."""
        return self.budget - self.total_cost()

    def check_budget(self):
        """Check if we are within budget. Raises RuntimeError if exceeded."""
        remaining = self.remaining_budget()
        if remaining < 0:
            raise RuntimeError(
                f"Budget exceeded! Total cost: ${self.total_cost():.4f}, "
                f"Budget: ${self.budget:.2f}, Over by: ${-remaining:.4f}"
            )

    def is_over_budget(self):
        """Return True if budget is exceeded."""
        return self.remaining_budget() < 0

    def print_summary(self):
        """Print a formatted cost summary."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("COST TRACKER SUMMARY")
        print("=" * 70)
        print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Budget:  ${self.budget:.2f}")
        print()

        print(f"  {'Model':<15s} | {'Calls':>7s} | {'In Tok':>10s} | {'Out Tok':>10s} | "
              f"{'Cost In':>8s} | {'Cost Out':>8s} | {'Total':>8s}")
        print("  " + "-" * 80)

        grand_total = 0.0
        for model_name, entry in sorted(self.usage.items()):
            cost_in = entry.get("cost_input", 0.0)
            cost_out = entry.get("cost_output", 0.0)
            cost_total = cost_in + cost_out
            grand_total += cost_total
            print(
                f"  {model_name:<15s} | {entry['calls']:>7,} | "
                f"{entry['input_tokens']:>10,} | {entry['output_tokens']:>10,} | "
                f"${cost_in:>7.4f} | ${cost_out:>7.4f} | ${cost_total:>7.4f}"
            )

        print("  " + "-" * 80)
        print(f"  {'TOTAL':<15s} | {'':>7s} | {'':>10s} | {'':>10s} | "
              f"{'':>8s} | {'':>8s} | ${grand_total:>7.4f}")
        print(f"  Remaining: ${self.remaining_budget():.4f}")

        if self.is_over_budget():
            print("  *** WARNING: OVER BUDGET ***")
        print()

    def to_dict(self):
        """Export tracker state as a JSON-serializable dict."""
        return {
            "budget": self.budget,
            "total_cost": self.total_cost(),
            "remaining": self.remaining_budget(),
            "elapsed_seconds": time.time() - self.start_time,
            "models": {
                name: {
                    "calls": entry["calls"],
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "cost_input": round(entry.get("cost_input", 0.0), 6),
                    "cost_output": round(entry.get("cost_output", 0.0), 6),
                    "cost_total": round(
                        entry.get("cost_input", 0.0) + entry.get("cost_output", 0.0), 6
                    ),
                }
                for name, entry in self.usage.items()
            },
        }

    def save(self, filepath):
        """Save tracker state to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Cost tracker saved: {filepath}")

    @classmethod
    def estimate_cost(cls, model_name, n_examples, avg_prompt_tokens=400, avg_completion_tokens=15):
        """Estimate total cost before running an experiment.

        Parameters
        ----------
        model_name : str
            Model name key for PRICING dict.
        n_examples : int
            Number of API calls to make.
        avg_prompt_tokens : int
            Average prompt tokens per call.
        avg_completion_tokens : int
            Average completion tokens per call.

        Returns
        -------
        dict
            Estimated costs.
        """
        pricing = PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        total_in = n_examples * avg_prompt_tokens
        total_out = n_examples * avg_completion_tokens
        cost_in = total_in * pricing["input"] / 1_000_000
        cost_out = total_out * pricing["output"] / 1_000_000
        return {
            "model": model_name,
            "n_examples": n_examples,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "cost_input": round(cost_in, 4),
            "cost_output": round(cost_out, 4),
            "cost_total": round(cost_in + cost_out, 4),
        }
