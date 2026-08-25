#!/bin/bash
## =====================================================================
## Denominator sweep + model/reasoning comparison on ARC.
##
## Serves ONE model per job and runs the `limit` method across several
## denominator caps, plus the other three methods once.
##
## Submit one job (default GLM-5.2):
##   sbatch run_sweep.sh
##
## Launch reasoning-level axis in parallel:
##   sbatch --export=ALL,MODEL=GLM-5.2-non-thinking  run_sweep.sh
##   sbatch --export=ALL,MODEL=GLM-5.2               run_sweep.sh
##   sbatch --export=ALL,MODEL=GLM-5.2-thinking-high run_sweep.sh
##
## BEFORE SUBMITTING, verify on a login node:
##   ls /common/data/models/ | grep -i glm     # confirm MODEL_PATH folder
##   sinfo -s                                   # confirm partition name
##   (and that ~/envs/numform venv has openai + sympy)
## =====================================================================

#SBATCH --job-name=numform_sweep
#SBATCH --account=llmsemantics
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=numform_sweep.%j.out
#SBATCH --error=numform_sweep.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tiennguyen@vt.edu

set -euo pipefail

## ---- model selection (override with --export=ALL,MODEL=...) ----
## SERVED_NAME is the API name your eval calls; MODEL_PATH is the on-disk folder.
SERVED_NAME="${MODEL:-GLM-5.2}"

## Map served name -> on-disk model folder. Thinking variants of GLM share the
## same weights and differ only by reasoning_effort at request time, so they all
## point at the same folder. Confirm folder names with: ls /common/data/models/
case "$SERVED_NAME" in
    GLM-5.2|GLM-5.2-non-thinking|GLM-5.2-thinking-high)
        MODEL_PATH="/common/data/models/zai-org--GLM-5.2-FP8" ;;
    gpt-oss-120b)
        MODEL_PATH="/common/data/models/openai--gpt-oss-120b" ;;
    DeepSeek-V4-Flash)
        MODEL_PATH="/common/data/models/deepseek-ai--DeepSeek-V4-Flash" ;;
    *)
        echo "Unknown model '$SERVED_NAME'. Add it to the case block."; exit 1 ;;
esac

## count GPUs allocated so tensor-parallel matches --gres
NUM_GPUS=$(echo "${SLURM_GPUS_ON_NODE:-1}")

PORT=8000
API_KEY="local-$(openssl rand -hex 8)"
BASE_URL="http://127.0.0.1:${PORT}/v1"

## ---- environment ----
module reset
module load vLLM
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

## ---- reasoning effort passed through to the eval (client reads REASONING_EFFORT) ----
case "$SERVED_NAME" in
    *non-thinking*)   REASONING_EFFORT="low" ;;
    *thinking-high*)  REASONING_EFFORT="high" ;;
    *)                REASONING_EFFORT="medium" ;;
esac
export REASONING_EFFORT

## ---- 1. launch vLLM server ----
echo "Serving $SERVED_NAME from $MODEL_PATH on $NUM_GPUS GPU(s), effort=$REASONING_EFFORT"
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len 8192 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    > vllm_server.${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

## ---- 2. wait for readiness ----
for i in $(seq 1 90); do
    curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "server ready after ~$((i*10))s"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "server died"; tail -60 vllm_server.${SLURM_JOB_ID}.log; exit 1; }
    sleep 10
done

## ---- 3. run the eval ----
export ARC_BASE_URL="$BASE_URL"
export ARC_API_KEY="$API_KEY"
export ARC_MODEL="$SERVED_NAME"

## the sweep: `limit` across denominator caps
for N in 10 100 1000 1000000; do
    echo "=== limit sweep, max_den=${N} ==="
    python -m numform_eval.runner \
        --methods limit --max-den "$N" \
        --repeats 3 \
        --out "results_${SERVED_NAME}_limit_den${N}_${SLURM_JOB_ID}.csv"
done

## the other three methods once (they ignore max_den)
echo "=== original / exact / nsimplify ==="
python -m numform_eval.runner \
    --methods original exact nsimplify \
    --repeats 3 \
    --out "results_${SERVED_NAME}_other_${SLURM_JOB_ID}.csv"

echo "Sweep complete for ${SERVED_NAME}."