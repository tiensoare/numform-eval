# numform-eval

Test how a model's numeric-reasoning accuracy changes when the **same numbers**
are presented in different forms:

| method | what it sends the model | role |
|---|---|---|
| `original` | plain decimal (`3.14159…`) | control |
| `exact` | `Fraction(str(x))` — lossless huge `p/q` | baseline (exact but unwieldy) |
| `limit` | `limit_denominator(N)` — clean bounded fraction | main knob (sweep N) |
| `nsimplify` | symbolic recovery (`pi`, `sqrt(2)`, `pi/4`), fraction fallback | symbolic arm |

Same tasks, same ground truth, four representations → compare accuracy and
token cost per representation and per task kind.

## Install

```bash
pip install -r requirements.txt
```

## Try it with no API key (fake model)

```bash
python -m numform_eval.runner --mock
```

Runs the whole pipeline (prompt build → "model" → parse → score → CSV + summary)
against a rigged mock, so you can verify everything before spending quota.
Writes to `results/results.csv` by default (see Output below).

## Run against ARC

Both ARC endpoints are OpenAI-compatible — pick one, set env vars, run.
Requires VT network or VPN.

**Option A — llm-api (persistent key, simplest):**
```bash
# key from https://llm.arc.vt.edu  (Settings > Account > API keys)
export ARC_API_KEY=sk-...
export ARC_BASE_URL=https://llm-api.arc.vt.edu/api/v1
export ARC_MODEL=gpt-oss-120b
python -m numform_eval.runner --repeats 3 --sleep 0.2
```
Limits: 10 concurrent requests, 8000 tokens / non-streaming request. Use
`--sleep` to stay well under the concurrency cap.

**Option B — OOD (dedicated session, no rate limits, best for big batches):**
```bash
# start the LLM app at https://ood.arc.vt.edu, copy its session URL + key
export ARC_API_KEY=<session-key>
export ARC_BASE_URL=<session-url-ending-in>/v1
export ARC_MODEL=<model you launched>
python -m numform_eval.runner --repeats 5
```
Note the 1-hour idle timeout — keep the batch moving.

## Run on ARC compute nodes with Slurm (recommended for real runs)

`run_arc_eval.sh` serves an open-source model with **vLLM on a GPU node** and
runs the eval against that local server — all in one job. No external endpoint,
no rate limits, fully reproducible for a paper.

```bash
# 1. edit run_arc_eval.sh: set --account to your allocation, confirm the
#    partition/GPU names for your cluster, and MODEL_PATH (see below)
sbatch run_arc_eval.sh
squeue -u $USER            # watch the queue
tail -f numform_eval.*.out # watch progress
seff <jobid>               # memory report after it finishes
```

Key ARC conventions baked into the script:
- Serves from `/common/data/models/` (not HF names) to avoid re-downloading —
  confirm the exact folder with `ls /common/data/models/`.
- `module load vLLM`, then `vllm serve <path> --served-model-name ... --api-key ...`.
- Default model **GLM-5.2** (ARC's strongest math/reasoning model). For
  `gpt-oss-120b` use 2 GPUs + `--tensor-parallel-size 2`.
- You must accept each model's HuggingFace terms with your HF account first.

`run_sweep.sh` does the denominator sweep (the core experiment) plus the other
three methods in one job. To compare models, copy it and change
`MODEL_PATH`/`SERVED_NAME` — which model best exploits symbolic input is part of
the finding, so run more than one.

**Want full-precision GLM-5.2 instead of quantized?** `/common/data/models`
only has quantized builds (`zai-org--GLM-5.2-FP8`, `nvidia--GLM-5.2-NVFP4`) —
full bf16 weights are ~1.3TB and don't fit on any realistic single-node
allocation. Use `run_sweep_api.sh` instead: it hits ARC's hosted
`llm-api.arc.vt.edu` endpoint (no GPU, no vLLM), needs `ARC_API_KEY` exported
before `sbatch`, and runs sequentially with `--sleep` since that endpoint is
explicitly not meant for batch/bulk use and caps concurrency at 10
requests/user — expect it to be much slower than the self-hosted path.

## Options

```
--out PATH        output CSV (default results.csv, written under results/;
                   a bare filename always lands in results/, a path with a
                   directory in it — e.g. --out other_dir/foo.csv — is used as-is)
--repeats N       trials per (task, method); >1 to average model nondeterminism
--max-den N       denominator cap for the `limit` method (sweep this!)
--methods ...     subset, e.g. --methods original limit nsimplify
--mock            use the fake model (no network/key)
--sleep S         seconds between requests (rate-limit friendliness)
```

## The denominator sweep (core experiment)

Vary the `limit` cap to find where clean-vs-precise best helps the model:

```bash
for N in 10 100 1000 1000000; do
  python -m numform_eval.runner --methods limit --max-den $N \
    --out results_limit_$N.csv --repeats 3   # -> results/results_limit_$N.csv
done
```

## Output

All results CSVs live under `results/` (created automatically). Tidy CSV,
one row per trial, with columns: model, task_id, kind, method,
repeat, prompt, response, parsed answer, truth, correct, abs/rel error,
parse_ok, prompt/completion tokens, latency, error. Load with pandas:

```python
import pandas as pd
df = pd.read_csv("results/results.csv")
df.groupby(["kind", "method"])["correct"].mean().unstack()
df.groupby("method")["completion_tokens"].mean()   # token cost per form
```

## Files

- `transforms.py` — the four number representations (+ `Number` provenance)
- `tasks.py` — problems with exact symbolic ground truth
- `client.py` — ARC OpenAI-compatible client (env-configured, retries)
- `scoring.py` — answer extraction + tolerance scoring (floats, fractions, symbols)
- `runner.py` — orchestration, CSV output, summary; `--mock` for dry runs
- `analyze.py` — combine/compare multiple results CSVs (e.g. original vs
  exact vs limit(N) vs nsimplify, or several `--max-den` sweeps at once):
  `python -m numform_eval.analyze results/results_*.csv`. Writes a report to
  `analysis/summary.txt` (no console output beyond the file path); add
  `--out combined.csv` to also save the merged rows to `analysis/combined.csv`.

## Extending

- **More tasks:** add `Task(...)` entries in `tasks.py`. Keep `ground_truth`
  symbolic so the reference answer never suffers truncation.
- **More constants for nsimplify:** edit `DEFAULT_CONSTS` in `transforms.py`.
- **Track reasoning steps / tokens:** `completion_tokens` is already logged;
  for reasoning-trace length, set `reasoning_effort` in `client.py` and inspect
  the response text.
- **Stronger vs smaller models:** just change `ARC_MODEL` and re-run to the
  same CSV schema, then compare.
