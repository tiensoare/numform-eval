"""
Experiment runner.

For each (task x method) it builds a prompt, queries the model, scores the
answer, and records a row. Output is a tidy CSV: one row per trial, ready for
pandas / plotting.

Usage (real ARC run):
    export ARC_API_KEY=sk-...           # from llm.arc.vt.edu or the OOD app
    export ARC_BASE_URL=https://llm-api.arc.vt.edu/api/v1
    export ARC_MODEL=gpt-oss-120b
    python -m numform_eval.runner --repeats 3    # writes to results/results.csv

Dry run with a fake model (no network, no key):
    python -m numform_eval.runner --mock

--out always controls the destination; relative paths without a directory
component (e.g. --out foo.csv) are placed under results/ automatically.
"""

import argparse
import csv
import math
import os
import time

RESULTS_DIR = "results"
from typing import Callable, List, Optional

from .tasks import Task, default_tasks
from .scoring import score
from .transforms import METHODS


FIELDS = [
    "model", "task_id", "kind", "method", "repeat",
    "prompt", "response", "answer_text", "parsed", "truth",
    "correct", "abs_err", "rel_err", "parse_ok",
    "prompt_tokens", "completion_tokens", "latency_s", "error",
]


def mock_model(prompt: str) -> dict:
    """
    Fake model for pipeline testing. Heuristic: if the prompt contains a
    symbolic form (pi, sqrt), 'solve' it more accurately; if it contains
    long decimals, introduce a small rounding error. This lets us see the
    experiment mechanics and roughly mimics the hypothesis.
    """
    import re
    import sympy as sp

    # crude: grab the math expression after "Compute" if present
    text = prompt.splitlines()[0]
    m = re.search(r"Compute (.+?)\.", text)
    val = None
    if m:
        expr_str = m.group(1).replace("squared", "**2")
        expr_str = re.sub(r"(.+)\s*\*\*2", r"(\1)**2", expr_str)
        try:
            val = float(sp.sympify(
                expr_str, locals={"pi": sp.pi, "sqrt": sp.sqrt}))
        except Exception:
            val = None
    if val is None:
        # comparison task or parse failure: just guess 1
        val = 1.0
    # decimals in prompt -> add noise; symbolic -> keep exact
    if re.search(r"\d\.\d{6,}", prompt):
        val = val * (1 + 4e-4)   # small systematic error for decimal inputs
    return {
        "text": f"Working...\nANSWER: {val:.6f}",
        "prompt_tokens": len(prompt) // 4,
        "completion_tokens": 12,
        "latency_s": 0.001,
        "error": None,
    }


def run(
    complete_fn: Callable[[str], dict],
    model_name: str,
    tasks: List[Task],
    methods: Optional[List[str]] = None,
    repeats: int = 1,
    max_den: int = 1000,
    out_path: str = "results.csv",
    sleep_between: float = 0.0,
):
    methods = methods or list(METHODS.keys())
    rows = []
    total = len(tasks) * len(methods) * repeats
    i = 0
    for task in tasks:
        truth = task.truth_float()
        for method in methods:
            # pass max_den only to the method that accepts it
            if method == "limit":
                prompt = task.build_prompt(method, max_den=max_den)
            else:
                prompt = task.build_prompt(method)
            for rep in range(repeats):
                i += 1
                res = complete_fn(prompt)
                sc = score(res["text"], truth, task.answer_tol)
                rows.append({
                    "model": model_name,
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "method": method,
                    "repeat": rep,
                    "prompt": prompt,
                    "response": res["text"],
                    "answer_text": sc["answer_text"],
                    "parsed": sc["parsed"],
                    "truth": truth,
                    "correct": sc["correct"],
                    "abs_err": sc["abs_err"],
                    "rel_err": sc["rel_err"],
                    "parse_ok": sc["parse_ok"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "latency_s": res["latency_s"],
                    "error": res["error"],
                })
                print(f"[{i}/{total}] {task.task_id:<20} {method:<10} "
                      f"correct={sc['correct']}")
                if sleep_between:
                    time.sleep(sleep_between)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    return rows


def summarize(rows):
    """Print accuracy by method and by (kind, method)."""
    from collections import defaultdict
    by_method = defaultdict(lambda: [0, 0])
    by_kind_method = defaultdict(lambda: [0, 0])
    for r in rows:
        by_method[r["method"]][1] += 1
        by_method[r["method"]][0] += int(bool(r["correct"]))
        key = (r["kind"], r["method"])
        by_kind_method[key][1] += 1
        by_kind_method[key][0] += int(bool(r["correct"]))

    print("\nAccuracy by method:")
    for m, (c, n) in sorted(by_method.items()):
        print(f"  {m:<10} {c}/{n} = {c/n:.1%}")

    print("\nAccuracy by kind x method:")
    for (k, m), (c, n) in sorted(by_kind_method.items()):
        print(f"  {k:<12} {m:<10} {c}/{n} = {c/n:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.csv",
                    help="output CSV; a bare filename (no directory part) "
                         f"is placed under {RESULTS_DIR}/")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-den", type=int, default=1000)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="subset of: original exact limit nsimplify")
    ap.add_argument("--mock", action="store_true",
                    help="use the fake model (no network / key needed)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between requests (respect rate limits)")
    args = ap.parse_args()

    out_path = args.out
    if not os.path.dirname(out_path):
        out_path = os.path.join(RESULTS_DIR, out_path)

    tasks = default_tasks()

    if args.mock:
        complete_fn = mock_model
        model_name = "mock"
    else:
        from .client import LLMClient, LLMConfig
        cfg = LLMConfig.from_env()
        client = LLMClient(cfg)
        complete_fn = client.complete
        model_name = cfg.model

    rows = run(
        complete_fn, model_name, tasks,
        methods=args.methods, repeats=args.repeats,
        max_den=args.max_den, out_path=out_path,
        sleep_between=args.sleep,
    )
    summarize(rows)


if __name__ == "__main__":
    main()
