#!/bin/bash
#SBATCH -J training_invrgbl
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH -w rtx8000
#SBATCH -o work_dirs/%x.%j.out

cd /lab/haoq_lab/cse12310504/InvRGBL
mkdir -p logs

source ~/.bashrc
conda activate InvRGBL

set -euo pipefail


 python -u tools/train.py \
    --config_file configs/invrgbl.yaml \
    --output_root ./work_dirs \
    --project invrgbl \
    --run_name waymo023_dyn_5cam \
    --resume_from work_dirs/invrgbl/waymo023_dyn_5cam/checkpoint_15000.pth \
    dataset=waymo/5cams \