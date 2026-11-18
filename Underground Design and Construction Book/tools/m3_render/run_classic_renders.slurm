#!/bin/bash
# Final stills of the book's three-dimensional figures on the M3 H100s.
# The folder on M3 holds render_classic_3d.py, render3d/ (scene modules), scenes.json and scenes.csv.
# One array task = one scene (task N = line N+1 of scenes.csv):
#   sbatch --array=1 run_classic_renders.slurm          (test one scene first)
#   sbatch --array=2-<N> run_classic_renders.slurm      (the rest)
# Output: out/<scene>.png (white background, ready for figures/render/ in the book).
#SBATCH --job-name=bookrender
#SBATCH --account=ok99
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -e
HERE="${SLURM_SUBMIT_DIR}"
BASE=/fs04/scratch2/ok99/qianbinz/blender
SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HERE/scenes.csv" | cut -d, -f1)
if [ -z "$SCENE" ]; then echo "no scene on line $((SLURM_ARRAY_TASK_ID + 1)) of scenes.csv"; exit 1; fi
if [ -f "$HERE/out/$SCENE.png" ]; then echo "$SCENE done"; exit 0; fi
export CUDA_CACHE_PATH="$BASE/cuda_cache" CUDA_CACHE_MAXSIZE=4294967296
echo "scene $SLURM_ARRAY_TASK_ID: $SCENE, ${RES:-3600x2400}, ${SAMPLES:-1024} samples, on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$HERE/out"
"$BASE/current/blender" -b --factory-startup -t "${SLURM_CPUS_PER_TASK}" \
    --python-exit-code 3 -P "$HERE/render_classic_3d.py" -- \
    --scene "$SCENE" --data "$HERE/scenes.json" --out "$HERE/out" \
    --samples "${SAMPLES:-1024}" --res "${RES:-3600x2400}" --device CUDA --save-blend
echo "done $SCENE"
