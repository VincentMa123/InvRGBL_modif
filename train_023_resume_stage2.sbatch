#!/bin/bash
#SBATCH -J training_ispbr_stage2
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

RUN_NAME=waymo023_dyn_1cam_ispbr
RUN_DIR=./work_dirs/invrgbl/${RUN_NAME}
DATA_ROOT=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training
SCENE_IDX=23
SCENE_ID=$(printf "%03d" ${SCENE_IDX})
START_TIMESTEP=0
END_TIMESTEP=50
FREEZE_STEP=15000
FINAL_STEP=30000
REGION_DIR=region_sam_reflectivity

# Override at submit time if your SAM checkpoint lives elsewhere:
# sbatch --export=ALL,SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth train_023_resume_stage2.sbatch
SAM_CHECKPOINT=${SAM_CHECKPOINT:-/lab/haoq_lab/cse12310504/InvRGBL/pretrained/sam_vit_h_4b8939.pth}
SAM_MODEL_TYPE=${SAM_MODEL_TYPE:-vit_h}
SAM_MAX_FRAMES=${SAM_MAX_FRAMES:-0}
FORCE_SAM=${FORCE_SAM:-0}
RESET_STAGE2_MATERIALS=${RESET_STAGE2_MATERIALS:-1}

STAGE1_FINAL=${RUN_DIR}/checkpoint_final.pth
STAGE1_CKPT=${RUN_DIR}/checkpoint_${FREEZE_STEP}.pth
REGION_PATH=${DATA_ROOT}/${SCENE_ID}/${REGION_DIR}


COMMON_OPTS=(
    dataset=waymo/1cams
    data.data_root=${DATA_ROOT}
    data.scene_idx=${SCENE_IDX}
    data.start_timestep=${START_TIMESTEP}
    data.end_timestep=${END_TIMESTEP}
)

if [[ "${FORCE_SAM}" == "1" ]] || ! compgen -G "${REGION_PATH}/*.npy" > /dev/null; then
    if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
        echo "Missing SAM checkpoint: ${SAM_CHECKPOINT}" >&2
        echo "Set SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth or put it at the default path." >&2
        exit 1
    fi

    echo "Precomputing SAM regions from ${STAGE1_CKPT}"
    python -u tools/precompute_reflectivity_sam_regions.py \
        --resume_from "${STAGE1_CKPT}" \
        --sam_checkpoint "${SAM_CHECKPOINT}" \
        --sam_model_type "${SAM_MODEL_TYPE}" \
        --split full \
        --max_frames "${SAM_MAX_FRAMES}" \
        --output_dir "${REGION_DIR}" \
        "${COMMON_OPTS[@]}"
else
    echo "Using existing SAM region maps in ${REGION_PATH}"
fi

RESUME_CKPT=${STAGE1_CKPT}

echo "Stage 2: resuming from ${RESUME_CKPT} with cached SAM region maps"
python -u tools/train.py \
    --config_file configs/invrgbl.yaml \
    --output_root ./work_dirs \
    --project invrgbl \
    --run_name ${RUN_NAME} \
    --resume_from "${RESUME_CKPT}" \
    "${COMMON_OPTS[@]}" \
    trainer.optim.num_iters=${FINAL_STEP} \
    data.pixel_source.load_region_maps=true \
    data.pixel_source.region_map_dir=${REGION_DIR} \
    trainer.losses.intensity.normalize_valid_pixels=true \
    trainer.losses.ref_neighborhood_smoothness.w=0.02
