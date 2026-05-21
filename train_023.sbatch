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

cd ~/project2/InvRGBL_modif
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
START_TIMESTEP=0
END_TIMESTEP=50
FREEZE_STEP=15000
FINAL_STEP=30000
REGION_DIR=region_sam_reflectivity_bgcontact

# Override at submit time if your SAM checkpoint lives elsewhere:
# sbatch --export=ALL,SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth train_023.sbatch
SAM_CHECKPOINT=${SAM_CHECKPOINT:-/lab/haoq_lab/cse12310504/InvRGBL/pretrained/sam_vit_h_4b8939.pth}
SAM_MODEL_TYPE=${SAM_MODEL_TYPE:-vit_h}
SAM_MAX_FRAMES=${SAM_MAX_FRAMES:-0}

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
    echo "Missing SAM checkpoint: ${SAM_CHECKPOINT}" >&2
    echo "Set SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth or edit train_023.sbatch." >&2
    exit 1
fi

COMMON_OPTS=(
    dataset=waymo/1cams
    data.data_root=${DATA_ROOT}
    data.scene_idx=${SCENE_IDX}
    data.start_timestep=${START_TIMESTEP}
    data.end_timestep=${END_TIMESTEP}
)

echo "Stage 1: training to freeze step ${FREEZE_STEP}"
python -u tools/train.py \
    --config_file configs/invrgbl.yaml \
    --output_root ./work_dirs \
    --project invrgbl \
    --run_name ${RUN_NAME} \
    "${COMMON_OPTS[@]}" \
    trainer.optim.num_iters=${FREEZE_STEP} \
    data.pixel_source.load_region_maps=false \
    trainer.losses.ref_region_consistency_albedo.w=0.0 \
    trainer.losses.intensity.stage1_w=0.3

STAGE1_FINAL=${RUN_DIR}/checkpoint_final.pth
STAGE1_CKPT=${RUN_DIR}/checkpoint_${FREEZE_STEP}.pth
if [[ ! -f "${STAGE1_FINAL}" ]]; then
    echo "Stage-1 checkpoint not found: ${STAGE1_FINAL}" >&2
    exit 1
fi

echo "Copying ${STAGE1_FINAL} to ${STAGE1_CKPT} for SAM region precomputation"
cp -f "${STAGE1_FINAL}" "${STAGE1_CKPT}"

echo "Precomputing SAM regions from ${STAGE1_CKPT}"
python -u tools/precompute_reflectivity_sam_regions.py \
    --resume_from "${STAGE1_CKPT}" \
    --sam_checkpoint "${SAM_CHECKPOINT}" \
    --sam_model_type "${SAM_MODEL_TYPE}" \
    --split full \
    --max_frames "${SAM_MAX_FRAMES}" \
    --output_dir "${REGION_DIR}" \
    "${COMMON_OPTS[@]}"

echo "Stage 2: resuming with cached SAM region maps"
python -u tools/train.py \
    --config_file configs/invrgbl.yaml \
    --output_root ./work_dirs \
    --project invrgbl \
    --run_name ${RUN_NAME} \
    --resume_from "${STAGE1_CKPT}" \
    "${COMMON_OPTS[@]}" \
    trainer.optim.num_iters=${FINAL_STEP} \
    data.pixel_source.load_region_maps=true \
    data.pixel_source.region_map_dir=${REGION_DIR} \
    trainer.render.dynamic_box_sun_visibility.enabled=true \
    trainer.render.dynamic_box_sun_visibility.receiver_static_only=true \
    trainer.render.dynamic_box_sun_visibility.receiver_dynamic_opacity_threshold=0.5 \
    trainer.render.dynamic_box_sun_visibility.skip_inside_boxes=true \
    trainer.render.dynamic_box_sun_visibility.shadow_strength=0.7 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_strength=0.7 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_receiver=background \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_height=1.0 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_softness=0.55 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_dynamic_opacity_threshold=0.8 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_apply_dynamic_opacity_threshold=0.8 \
    trainer.render.dynamic_box_sun_visibility.contact_shadow_apply_depth_tolerance=0.5 \
    trainer.losses.intensity.stage2_w=0.2 \
    trainer.losses.ref_neighborhood_smoothness.w=0.05 \
    trainer.losses.ref_region_consistency_albedo.w=0.05
