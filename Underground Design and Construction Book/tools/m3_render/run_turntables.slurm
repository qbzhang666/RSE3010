#!/bin/bash
# 360 degree turntables of selected scenes, for lectures and the book's online material.
# One array task = one scene (task N = line N+1 of turntables.csv). Frames are skipped if
# they already exist, so a task that hits the time limit is simply resubmitted.
#   sbatch --array=1 run_turntables.slurm              (test one scene first)
#   sbatch --array=2-<N> run_turntables.slurm
# Output: out/<scene>_turn/f_0001.png ... ; encode on the PC with ffmpeg.
#SBATCH --job-name=bookturn
#SBATCH --account=ok99
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -e
HERE="${SLURM_SUBMIT_DIR}"
BASE=/fs04/scratch2/ok99/qianbinz/blender
SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HERE/turntables.csv" | cut -d, -f1)
if [ -z "$SCENE" ]; then echo "no scene on line $((SLURM_ARRAY_TASK_ID + 1)) of turntables.csv"; exit 1; fi
export CUDA_CACHE_PATH="$BASE/cuda_cache" CUDA_CACHE_MAXSIZE=4294967296
echo "turntable $SLURM_ARRAY_TASK_ID: $SCENE, ${FRAMES:-240} frames, ${RES:-1920x1080}, ${SAMPLES:-256} samples, on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$HERE/out"
"$BASE/current/blender" -b --factory-startup -t "${SLURM_CPUS_PER_TASK}" \
    --python-exit-code 3 -P "$HERE/render_classic_3d.py" -- \
    --scene "$SCENE" --data "$HERE/scenes.json" --out "$HERE/out" --turntable "${FRAMES:-240}" \
    --samples "${SAMPLES:-256}" --res "${RES:-1920x1080}" --device CUDA
echo "done $SCENE"
