#!/bin/bash

#SBATCH --job-name=per_dqn
#SBATCH --output=/usr/users/rl_course_26/rl_course_26_27/rl-highway/results/logs/dce/per_dqn/slurm_%j.out
#SBATCH --error=/usr/users/rl_course_26/rl_course_26_27/rl-highway/results/logs/dce/per_dqn/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_prod_long

set -euo pipefail

REPO_DIR="/usr/users/rl_course_26/rl_course_26_27/rl-highway"
LOG_DIR="$REPO_DIR/results/logs/dce/per_dqn"

source "$HOME/my_rl_venv/bin/activate"
cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

echo "Start: $(date)"
echo "Host : $(hostname)"
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python core_task/train_dqn_per.py "$@"

echo "End: $(date)"
