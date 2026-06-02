#!/bin/bash
#SBATCH -J relight_spotlights
#SBATCH --chdir=/lab/haoq_lab/cse12312032/InvRGBL_modif
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=titanrtx02,titanrtx03,titanv01
#SBATCH -w rtx8000
#SBATCH -o /dev/null
#SBATCH -e /dev/null

PROJECT_ROOT=${PROJECT_ROOT:-/lab/haoq_lab/cse12312032/InvRGBL_modif}
cd "${PROJECT_ROOT}"
mkdir -p logs work_dirs configs

JOB_LOG_ID=${SLURM_JOB_ID:-manual}
exec > "${PROJECT_ROOT}/work_dirs/relight_spotlights.${JOB_LOG_ID}.out" \
     2> "${PROJECT_ROOT}/work_dirs/relight_spotlights.${JOB_LOG_ID}.err"
echo "Started relight_spotlights job ${JOB_LOG_ID} on $(hostname)"
echo "Work dir: $(pwd)"

source ~/.bashrc
conda activate InvRGBL

unset CUDA_HOME
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
set -euo pipefail

RUN_NAME=${RUN_NAME:-waymo023_dyn_1cam_ispbr}
CHECKPOINT=${CHECKPOINT:-work_dirs/invrgbl/${RUN_NAME}/checkpoint_final_old.pth}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/waymo/processed/training}
SCENE_IDX=${SCENE_IDX:-23}
START_TIMESTEP=${START_TIMESTEP:-0}
END_TIMESTEP=${END_TIMESTEP:-50}
MAX_FRAMES=${MAX_FRAMES:-50}
PRESET=${PRESET:-night}
SPOTLIGHTS=${SPOTLIGHTS:-configs/spotlights_023_template.json}
OUTPUT_NAME=${OUTPUT_NAME:-night_spotlights_darker_test}
DUMP_TEMPLATE=${DUMP_TEMPLATE:-0}
ENV_DIFFUSE_SCALE=${ENV_DIFFUSE_SCALE:-0.012}
ENV_SPECULAR_SCALE=${ENV_SPECULAR_SCALE:-0.06}
ENV_DIFFUSE_MODE=${ENV_DIFFUSE_MODE:-neutral}
MIN_ROUGHNESS=${MIN_ROUGHNESS:-0.25}
SKY_SCALE=${SKY_SCALE:-0.02}
EXPOSURE_SCALE=${EXPOSURE_SCALE:-0.28}

echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Spotlights: ${SPOTLIGHTS}"

if [[ "${DUMP_TEMPLATE}" == "1" ]]; then
    python -u tools/relight.py \
        --resume_from "${CHECKPOINT}" \
        --preset "${PRESET}" \
        --split none \
        --dump_spotlight_template "${SPOTLIGHTS}" \
        data.data_root="${DATA_ROOT}" \
        data.scene_idx="${SCENE_IDX}" \
        data.start_timestep="${START_TIMESTEP}" \
        data.end_timestep="${END_TIMESTEP}"
    exit 0
fi

if [[ ! -f "${SPOTLIGHTS}" ]]; then
    echo "Missing spotlight config: ${SPOTLIGHTS}" >&2
    echo "Run: sbatch --export=ALL,DUMP_TEMPLATE=1 relight_023.sbatch, then edit ${SPOTLIGHTS}" >&2
    exit 1
fi

python -u tools/relight.py \
    --resume_from "${CHECKPOINT}" \
    --name "${OUTPUT_NAME}" \
    --preset "${PRESET}" \
    --spotlights "${SPOTLIGHTS}" \
    --split full \
    --max_frames "${MAX_FRAMES}" \
    --save_images \
    --force_dielectric \
    --min_roughness "${MIN_ROUGHNESS}" \
    --sky_scale "${SKY_SCALE}" \
    --exposure_scale "${EXPOSURE_SCALE}" \
    --disable_affine \
    --dynamic_box_sun_visibility \
    --dynamic_box_skip_inside_boxes \
    --dynamic_box_receiver_static_only \
    --dynamic_box_receiver_dynamic_opacity_threshold 0.5 \
    --dynamic_box_shadow_strength 0.7 \
    --dynamic_box_contact_shadow_strength 0.55 \
    --dynamic_box_contact_shadow_receiver background \
    --dynamic_box_contact_shadow_height 0.8 \
    --dynamic_box_contact_shadow_softness 0.8 \
    --dynamic_box_contact_shadow_dynamic_opacity_threshold 0.8 \
    --dynamic_box_contact_shadow_apply_dynamic_opacity_threshold 1.1 \
    --dynamic_box_contact_shadow_apply_depth_tolerance 0.5 \
    --dynamic_box_contact_shadow_support_near_dynamic \
    --dynamic_box_contact_shadow_support_dynamic_opacity_threshold 0.08 \
    --dynamic_box_contact_shadow_support_dilate_pixels 36 \
    --dynamic_box_layer_separated_pbr \
    --dynamic_box_layer_dynamic_depth_margin 0.25 \
    --env_diffuse_scale "${ENV_DIFFUSE_SCALE}" \
    --env_specular_scale "${ENV_SPECULAR_SCALE}" \
    --env_diffuse_mode "${ENV_DIFFUSE_MODE}" \
    --keys rgbs,rendered_pbr,rendered_spotlight,rendered_pbr_layered,rendered_pbr_background,rendered_pbr_dynamic,rendered_pbr_dynamic_composite_mask,rendered_albedos,rendered_sun_visibility,rendered_dynamic_box_sun_visibility,rendered_dynamic_box_contact_shadow,rendered_dynamic_box_contact_shadow_raw,rendered_dynamic_box_contact_apply_mask,rendered_dynamic_box_contact_support_mask,rendered_dynamic_opacity,depths \
    data.data_root="${DATA_ROOT}" \
    data.scene_idx="${SCENE_IDX}" \
    data.start_timestep="${START_TIMESTEP}" \
    data.end_timestep="${END_TIMESTEP}"
