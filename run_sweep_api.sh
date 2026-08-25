#!/bin/bash
## =====================================================================
## Denominator sweep + reasoning-effort comparison against ARC's hosted
## LLM API (llm-api.arc.vt.edu), NOT a self-hosted checkpoint.
##
## Use this instead of run_sweep.sh when you want full-precision GLM-5.2:
## the only on-disk copies in /common/data/models are quantized
## (zai-org--GLM-5.2-FP8, nvidia--GLM-5.2-NVFP4) and full bf16 weights
## (~1.3TB) don't fit on any realistic single-node allocation here.
## No GPU needed — this job just makes HTTP calls.
##
## Submit one job (default effort=medium):
##   sbatch run_sweep_api.sh
##
## Launch reasoning-level axis in parallel:
##   sbatch --export=ALL,REASONING=low    run_sweep_api.sh
##   sbatch --export=ALL,REASONING=medium run_sweep_api.sh
##   sbatch --export=ALL,REASONING=high   run_sweep_api.sh
##
## BEFORE SUBMITTING:
##   export ARC_API_KEY=sk-...   # from https://llm.arc.vt.edu (Settings > Account > API keys)
##   (compute nodes reach llm-api.arc.vt.edu directly; no VPN needed from ARC)
##
## NOTE: ARC's docs say llm-api.arc.vt.edu is NOT for batch/bulk use and
## caps concurrency at 10 requests/user. runner.py issues requests
## strictly one at a time, and --sleep below adds extra spacing, so this
## stays well under that cap — but expect it to run much slower than the
## self-hosted vLLM path since it's a shared service.
## =====================================================================

#SBATCH --job-name=numform_sweep_api
#SBATCH --account=llmsemantics
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=numform_sweep_api.%j.out
#SBATCH --error=numform_sweep_api.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tiennguyen@vt.edu

set -euo pipefail

if [ -z "${ARC_API_KEY:-}" ]; then
    echo "ERROR: ARC_API_KEY not set. Export it before submitting:"
    echo "  export ARC_API_KEY=sk-...   # from https://llm.arc.vt.edu"
    exit 1
fi

export ARC_BASE_URL="${ARC_BASE_URL:-https://llm-api.arc.vt.edu/api/v1}"
export ARC_MODEL="GLM-5.2"
export REASONING_EFFORT="${REASONING:-medium}"
SERVED_NAME="GLM-5.2-${REASONING_EFFORT}"

## ---- environment ----
## Use your own venv with openai + sympy (create once beforehand):
##   python -m venv ~/envs/numform && source ~/envs/numform/bin/activate && pip install openai sympy
## If the venv doesn't exist, fall back to a --user install.
if [ -f "$HOME/envs/numform/bin/activate" ]; then
    source "$HOME/envs/numform/bin/activate"
else
    echo "WARN: ~/envs/numform not found; falling back to pip --user"
    pip install --quiet --user openai sympy || true
fi

cd "$SLURM_SUBMIT_DIR"

echo "Querying $ARC_MODEL via $ARC_BASE_URL, effort=$REASONING_EFFORT"

## ---- the sweep: `limit` across denominator caps ----
for N in 10 100 1000 1000000; do
    echo "=== limit sweep, max_den=${N} ==="
    python -m numform_eval.runner \
        --methods limit --max-den "$N" \
        --repeats 3 --sleep 0.3 \
        --out "results_${SERVED_NAME}_limit_den${N}_${SLURM_JOB_ID}.csv"
done

## ---- the other three methods once (they ignore max_den) ----
echo "=== original / exact / nsimplify ==="
python -m numform_eval.runner \
    --methods original exact nsimplify \
    --repeats 3 --sleep 0.3 \
    --out "results_${SERVED_NAME}_other_${SLURM_JOB_ID}.csv"

echo "Sweep complete for ${SERVED_NAME} (via ARC LLM API)."
