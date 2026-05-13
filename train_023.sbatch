#!/bin/bash
#SBATCH -J training_ispbr
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=titanrtx02,titanrtx03,titanv01
#SBATCH -w rtx8000
#SBATCH -o work_dirs/%x.%j.out
#SBATCH -e work_dirs/%x.%j.err

cd /lab/haoq_lab/cse12310504/InvRGBL
mkdir -p logs work_dirs

source ~/.bashrc
conda activate InvRGBL

# Fix broken CUDA_HOME on some nodes
unset CUDA_HOME

set -euo pipefail

# Debug: verify environment
which python
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"

# Train image-space PBR from scratch
python -u tools/train.py \
    --config_file configs/invrgbl.yaml \
    --output_root ./work_dirs \
    --project invrgbl \
    --run_name waymo023_dyn_1cam_ispbr \
    dataset=waymo/1cams \
    data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50
