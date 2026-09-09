#!/bin/bash
#SBATCH -A <YOUR_SLURM_ACCOUNT>
#SBATCH -p normal
#SBATCH -J Qwen3-Next-80B-A3B-Instruct
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --gpus=4
#SBATCH --output=log/xes_opt_%x_seed42_%j.out
#SBATCH --error=log/xes_opt_%x_seed42_%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${PROJECT_DIR}/log"
export PROJECT_DIR

cd "$PROJECT_DIR"

# <PATH_TO_YOUR_EDF_FILE>.toml: CSCS uenv/EDF container spec with vLLM installed
srun --environment=<PATH_TO_YOUR_EDF_FILE>.toml bash -lc '

set -euo pipefail

echo "-------------------- System Info --------------------"
python3 -V
python3 -c "import torch; print('"'"'torch'"'"', torch.__version__); print('"'"'cuda'"'"', torch.version.cuda)"
python3 -c "import vllm; print('"'"'vllm'"'"', vllm.__version__)"
echo "-------------------- System Info End --------------------"

RESULT_DIR="${PROJECT_DIR}/results_xes_opt"

seed=42

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_TOKEN="<YOUR_HF_TOKEN>"

cd "${PROJECT_DIR}"

for model_name in "Qwen3-Next-80B-A3B-Instruct"; do
  mkdir -p "${RESULT_DIR}/${model_name}-seed${seed}"
  bash ./sh/eval_xes_opt.sh "qwen25-math-cot" "Qwen/${model_name}" "${RESULT_DIR}/${model_name}-seed${seed}" "${seed}"
done

'
