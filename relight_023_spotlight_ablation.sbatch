#!/bin/bash
#SBATCH -J relight_ablation
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
mkdir -p work_dirs

JOB_LOG_ID=${SLURM_JOB_ID:-manual}
exec > "${PROJECT_ROOT}/work_dirs/relight_spotlight_ablation.${JOB_LOG_ID}.out" \
     2> "${PROJECT_ROOT}/work_dirs/relight_spotlight_ablation.${JOB_LOG_ID}.err"

echo "Started relight spotlight ablation job ${JOB_LOG_ID} on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"

source ~/.bashrc
conda activate InvRGBL

unset CUDA_HOME
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
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

ENV_DIFFUSE_SCALE=${ENV_DIFFUSE_SCALE:-0.012}
ENV_SPECULAR_SCALE=${ENV_SPECULAR_SCALE:-0.06}
ENV_DIFFUSE_MODE=${ENV_DIFFUSE_MODE:-neutral}
MIN_ROUGHNESS=${MIN_ROUGHNESS:-0.25}
SKY_SCALE=${SKY_SCALE:-0.02}
EXPOSURE_SCALE=${EXPOSURE_SCALE:-0.28}

COMMON_KEYS=rgbs,rendered_pbr,rendered_spotlight,rendered_pbr_layered,rendered_pbr_background,rendered_pbr_dynamic,rendered_albedos,rendered_sun_visibility,depths

run_relight_case() {
    local output_name=$1
    local spotlight_config=$2

    echo "Running ${output_name}"
    echo "Spotlight config: ${spotlight_config}"

    python -u tools/relight.py \
        --resume_from "${CHECKPOINT}" \
        --split full \
        --max_frames "${MAX_FRAMES}" \
        --name "${output_name}" \
        --preset "${PRESET}" \
        --spotlights "${spotlight_config}" \
        --keys "${COMMON_KEYS}" \
        --env_diffuse_scale "${ENV_DIFFUSE_SCALE}" \
        --env_specular_scale "${ENV_SPECULAR_SCALE}" \
        --env_diffuse_mode "${ENV_DIFFUSE_MODE}" \
        --force_dielectric \
        --min_roughness "${MIN_ROUGHNESS}" \
        --sky_scale "${SKY_SCALE}" \
        --exposure_scale "${EXPOSURE_SCALE}" \
        --disable_affine \
        --dynamic_box_sun_visibility \
        --save_images \
        data.data_root="${DATA_ROOT}" \
        data.scene_idx="${SCENE_IDX}" \
        data.start_timestep="${START_TIMESTEP}" \
        data.end_timestep="${END_TIMESTEP}"
}

run_relight_case night_ablation_corner_car_only configs/spotlights_023_corner_car_only.json
run_relight_case night_ablation_ego_only configs/spotlights_023_ego_only.json
run_relight_case night_ablation_all configs/spotlights_023_all.json

echo "Finished relight spotlight ablation job ${JOB_LOG_ID}"
