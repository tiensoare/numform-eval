#!/bin/bash
## =====================================================================
## numform-eval on ARC: serve an open-source model with vLLM on a GPU
## node, then run the number-representation evaluation against it, all
## inside one Slurm job. No external API, no rate limits, fully reproducible.
##
## Submit:   sbatch run_arc_eval.sh
## Watch:    squeue -u $USER        (and tail the .out file)
## Memory report after: seff <jobid>
## =====================================================================

## ---- Job parameters (EDIT the account; check partition for your cluster) ----
#SBATCH --job-name=numform-eval
#SBATCH --account=llmsemantics        # <-- REQUIRED: your allocation name
#SBATCH --partition=a100_normal_q           # TinkerCliffs A100 queue; see notes below
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1                    # 1 A100-80GB; bump to 2 for a 120B model
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00                    # 1 day; raise for big sweeps
#SBATCH --output=numform-eval.%j.out
#SBATCH --error=numform-eval.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tiennguyen@vt.edu

set -euo pipefail

## ---- Model selection --------------------------------------------------------
## Best default for numerical/math reasoning per ARC's own description.
## Serve from /common/data/models (saves storage, no re-download).
## Confirm the exact on-disk folder name with:  ls /common/data/models/
MODEL_PATH="/common/data/models/zai-org--GLM-5.2-FP8"
SERVED_NAME="GLM-5.2"

## A100-80GB holds GLM-5.2 comfortably. For gpt-oss-120b use 2 GPUs:
##   #SBATCH --gres=gpu:a100:2   and add  --tensor-parallel-size 2  below.

## ---- Server config ----------------------------------------------------------
PORT=8000
API_KEY="local-$(openssl rand -hex 8)"      # ephemeral key, this job only
BASE_URL="http://127.0.0.1:${PORT}/v1"
MAX_MODEL_LEN=8192

## ---- Environment ------------------------------------------------------------
module reset
module load vLLM
## Your eval deps (openai, sympy). Adjust to your env manager if needed:
##   source ~/envs/numform/bin/activate
pip install --quiet --user openai sympy || true

cd "$SLURM_SUBMIT_DIR"

## ---- 1. Launch the vLLM server in the background ----------------------------
echo "Starting vLLM server for ${SERVED_NAME} ..."
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    > vllm_server.${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

## Make sure we always kill the server when the job ends
cleanup() {
    echo "Shutting down vLLM server (PID ${VLLM_PID}) ..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

## ---- 2. Wait until the server is ready --------------------------------------
echo "Waiting for server to become ready ..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is up after ~$((i*10))s."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died during startup. See vllm_server.${SLURM_JOB_ID}.log"
        tail -50 "vllm_server.${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
done

## ---- 3. Run the evaluation against the local server -------------------------
export ARC_BASE_URL="$BASE_URL"
export ARC_API_KEY="$API_KEY"
export ARC_MODEL="$SERVED_NAME"

echo "Running numform-eval ..."
python -m numform_eval.runner \
    --out "results_${SERVED_NAME}_${SLURM_JOB_ID}.csv" \
    --repeats 3 \
    --max-den 1000

echo "Done. Results in results/results_${SERVED_NAME}_${SLURM_JOB_ID}.csv"

## ---- Notes ------------------------------------------------------------------
## * Partition/GPU names vary by cluster. On TinkerCliffs the GPU queues use
##   A100-80GB and H200 nodes. Verify names with:  sinfo -s   or the ARC docs.
##   If a100_normal_q is wrong for your allocation, check `scontrol show partition`.
## * If your model isn't in /common/data/models, request it via 4help, or
##   download to your own space and point MODEL_PATH there.
## * You must accept each model's HuggingFace terms with your HF account first.
## * For the denominator sweep, see run_sweep.sh.
