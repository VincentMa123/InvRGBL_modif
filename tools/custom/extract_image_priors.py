import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PriorJob:
    scene: str
    frame_stem: str
    input_image: Path
    output_normal_npy: Path
    output_albedo_jpg: Path
    output_roughness_jpg: Path

    @property
    def frame(self) -> str:
        return self.frame_stem.split("_")[0]

    @property
    def camera(self) -> str:
        return self.frame_stem.split("_")[1]

    def format_map(self) -> Dict[str, str]:
        return {
            "scene": self.scene,
            "frame_stem": self.frame_stem,
            "frame": self.frame,
            "camera": self.camera,
            "input_image": str(self.input_image),
            "scene_dir": str(self.input_image.parent.parent),
            "output_normal_npy": str(self.output_normal_npy),
            "output_albedo_jpg": str(self.output_albedo_jpg),
            "output_roughness_jpg": str(self.output_roughness_jpg),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 prior extraction scaffold for GeoWizard normals and RGB-X materials"
    )
    parser.add_argument(
        "--data_root",
        default="data/waymo/processed/training",
        help="Root directory of processed scene folders",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help="Optional scene ids to process, e.g. 023 114 327",
    )
    parser.add_argument(
        "--normal_command",
        default=None,
        help=(
            "Shell command template to generate a normal .npy file. "
            "Available fields: {input_image}, {output_normal_npy}, {scene}, {frame}, {camera}, {frame_stem}, {scene_dir}"
        ),
    )
    parser.add_argument(
        "--material_command",
        default=None,
        help=(
            "Shell command template to generate albedo and roughness JPGs. "
            "Available fields: {input_image}, {output_albedo_jpg}, {output_roughness_jpg}, {scene}, {frame}, {camera}, {frame_stem}, {scene_dir}"
        ),
    )
    parser.add_argument(
        "--normal_preset",
        choices=["geowizard", "geowizard_v2"],
        default=None,
        help="Built-in batch normal extractor preset",
    )
    parser.add_argument(
        "--material_preset",
        choices=["rgbx"],
        default=None,
        help="Built-in batch material extractor preset",
    )
    parser.add_argument(
        "--geowizard_root",
        default="GeoWizard",
        help="Path to the cloned GeoWizard repository",
    )
    parser.add_argument(
        "--rgbx_root",
        default="rgbx",
        help="Path to the cloned RGB-X repository",
    )
    parser.add_argument(
        "--normal_python",
        default="python",
        help="Python executable used for GeoWizard inference",
    )
    parser.add_argument(
        "--material_python",
        default="python",
        help="Python executable used for RGB-X inference",
    )
    parser.add_argument(
        "--geowizard_domain",
        default="outdoor",
        choices=["indoor", "outdoor", "object"],
        help="GeoWizard domain flag for Waymo images",
    )
    parser.add_argument(
        "--geowizard_ensemble_size",
        type=int,
        default=3,
        help="GeoWizard ensemble size",
    )
    parser.add_argument(
        "--geowizard_denoise_steps",
        type=int,
        default=10,
        help="GeoWizard denoising steps",
    )
    parser.add_argument(
        "--geowizard_seed",
        type=int,
        default=0,
        help="GeoWizard random seed",
    )
    parser.add_argument(
        "--geowizard_half_precision",
        action="store_true",
        help="Run GeoWizard with --half_precision",
    )
    parser.add_argument(
        "--rgbx_inference_steps",
        type=int,
        default=50,
        help="RGB-X denoising steps",
    )
    parser.add_argument(
        "--rgbx_seed",
        type=int,
        default=0,
        help="RGB-X random seed",
    )
    parser.add_argument(
        "--rgbx_device",
        default="cuda",
        help="Device passed to RGB-X wrapper, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip jobs whose requested outputs already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write the manifest without executing them",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads used to launch external commands",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of image jobs for testing; 0 means no limit",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional JSONL manifest path; defaults to <data_root>/prior_jobs.jsonl",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop as soon as one external command fails",
    )
    return parser.parse_args()


def validate_args(args):
    if args.normal_command and args.normal_preset:
        raise ValueError("Use either --normal_command or --normal_preset, not both")
    if args.material_command and args.material_preset:
        raise ValueError("Use either --material_command or --material_preset, not both")
    if (
        args.normal_command is None
        and args.material_command is None
        and args.normal_preset is None
        and args.material_preset is None
    ):
        raise ValueError(
            "Provide at least one of --normal_command, --material_command, --normal_preset, or --material_preset"
        )


def list_scene_dirs(data_root: Path, scene_ids: Optional[List[str]]) -> List[Path]:
    if scene_ids is None:
        return sorted([path for path in data_root.iterdir() if path.is_dir()])
    normalized = {str(scene).zfill(3) for scene in scene_ids}
    return [data_root / scene for scene in sorted(normalized) if (data_root / scene).is_dir()]


def build_jobs(scene_dir: Path) -> List[PriorJob]:
    image_dir = scene_dir / "images"
    if not image_dir.exists():
        return []

    normal_dir = scene_dir / "normals" / "normal_npy"
    albedo_dir = scene_dir / "albedo_rgbx"
    roughness_dir = scene_dir / "rough_rgbx"
    normal_dir.mkdir(parents=True, exist_ok=True)
    albedo_dir.mkdir(parents=True, exist_ok=True)
    roughness_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[PriorJob] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        frame_stem = image_path.stem
        jobs.append(
            PriorJob(
                scene=scene_dir.name,
                frame_stem=frame_stem,
                input_image=image_path,
                output_normal_npy=normal_dir / f"{frame_stem}_pred.npy",
                output_albedo_jpg=albedo_dir / f"{frame_stem}.jpg",
                output_roughness_jpg=roughness_dir / f"{frame_stem}.jpg",
            )
        )
    return jobs


def should_skip(job: PriorJob, args) -> bool:
    need_normal = args.normal_command is not None and not job.output_normal_npy.exists()
    need_material = args.material_command is not None and (
        not job.output_albedo_jpg.exists() or not job.output_roughness_jpg.exists()
    )
    if not args.skip_existing:
        return False
    return not need_normal and not need_material


def write_manifest(manifest_path: Path, jobs: Iterable[PriorJob]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.format_map()) + "\n")


def run_command(template: str, job: PriorJob, label: str, dry_run: bool) -> Optional[str]:
    command = template.format(**job.format_map())
    print(f"[{label}] {job.scene}/{job.frame_stem}: {command}")
    if dry_run:
        return None
    completed = subprocess.run(command, shell=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} command failed for {job.scene}/{job.frame_stem}\n"
            f"Command: {command}\n"
            f"Exit code: {completed.returncode}"
        )
    return None


def parse_command(command: str) -> List[str]:
    return shlex.split(command)


def enable_live_conda_output(command: List[str]) -> List[str]:
    if len(command) >= 2 and command[0] == "conda" and command[1] == "run":
        if "--no-capture-output" not in command and "--live-stream" not in command:
            return [command[0], command[1], "--no-capture-output", *command[2:]]
    return command


def run_process(
    command: List[str],
    cwd: Optional[Path],
    label: str,
    dry_run: bool,
    extra_env: Optional[Dict[str, str]] = None,
):
    command = enable_live_conda_output(command)
    pretty = " ".join(shlex.quote(part) for part in command)
    cwd_text = f" (cwd={cwd})" if cwd is not None else ""
    print(f"[{label}] {pretty}{cwd_text}", flush=True)
    if dry_run:
        return
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"Command: {pretty}\n"
            f"cwd: {cwd}\n"
            f"Exit code: {completed.returncode}"
        )


def link_or_copy_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def copy_if_exists(source: Path, target: Path):
    if not source.exists():
        raise FileNotFoundError(f"Expected output missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def group_jobs_by_scene(jobs: Iterable[PriorJob]) -> Dict[str, List[PriorJob]]:
    scene_jobs: Dict[str, List[PriorJob]] = {}
    for job in jobs:
        scene_jobs.setdefault(job.scene, []).append(job)
    return scene_jobs


def scene_jobs_needing_normals(jobs: List[PriorJob], args) -> List[PriorJob]:
    if args.normal_preset is None:
        return []
    if not args.skip_existing:
        return jobs
    return [job for job in jobs if not job.output_normal_npy.exists()]


def scene_jobs_needing_materials(jobs: List[PriorJob], args) -> List[PriorJob]:
    if args.material_preset is None:
        return []
    if not args.skip_existing:
        return jobs
    return [
        job
        for job in jobs
        if not job.output_albedo_jpg.exists() or not job.output_roughness_jpg.exists()
    ]


def stage_scene_inputs(jobs: List[PriorJob], temp_root: Path) -> Path:
    input_dir = temp_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        link_or_copy_file(job.input_image, input_dir / job.input_image.name)
    return input_dir


def run_geowizard_scene(scene_dir: Path, jobs: List[PriorJob], args):
    if not jobs:
        return
    geowizard_root = Path(args.geowizard_root).resolve()
    geowizard_workdir = geowizard_root / "geowizard"
    script_name = "run_infer_v2.py" if args.normal_preset == "geowizard_v2" else "run_infer.py"
    with tempfile.TemporaryDirectory(prefix=f"geowizard_{scene_dir.name}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_input_dir = stage_scene_inputs(jobs, temp_dir)
        staged_output_dir = temp_dir / "output"
        command = parse_command(args.normal_python) + [
            script_name,
            "--input_dir",
            str(staged_input_dir),
            "--output_dir",
            str(staged_output_dir),
            "--ensemble_size",
            str(args.geowizard_ensemble_size),
            "--denoise_steps",
            str(args.geowizard_denoise_steps),
            "--seed",
            str(args.geowizard_seed),
            "--domain",
            args.geowizard_domain,
        ]
        if args.geowizard_half_precision:
            command.append("--half_precision")
        run_process(
            command,
            geowizard_workdir,
            f"normal:{scene_dir.name}",
            args.dry_run,
            extra_env={"PYTHONPATH": str(geowizard_workdir)},
        )
        if args.dry_run:
            return
        staged_normal_dir = staged_output_dir / "normal_npy"
        for job in jobs:
            copy_if_exists(staged_normal_dir / f"{job.frame_stem}_pred.npy", job.output_normal_npy)


def run_rgbx_scene(scene_dir: Path, jobs: List[PriorJob], args):
    if not jobs:
        return
    wrapper_path = Path(__file__).resolve().parent / "run_rgbx_batch.py"
    with tempfile.TemporaryDirectory(prefix=f"rgbx_{scene_dir.name}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_input_dir = stage_scene_inputs(jobs, temp_dir)
        command = parse_command(args.material_python) + [
            str(wrapper_path),
            "--rgbx_root",
            str(Path(args.rgbx_root).resolve()),
            "--input_dir",
            str(staged_input_dir),
            "--output_albedo_dir",
            str(scene_dir / "albedo_rgbx"),
            "--output_roughness_dir",
            str(scene_dir / "rough_rgbx"),
            "--inference_steps",
            str(args.rgbx_inference_steps),
            "--seed",
            str(args.rgbx_seed),
            "--device",
            args.rgbx_device,
        ]
        if args.skip_existing:
            command.append("--skip_existing")
        run_process(command, Path(__file__).resolve().parents[1], f"material:{scene_dir.name}", args.dry_run)


def process_scene(scene_dir: Path, jobs: List[PriorJob], args):
    normal_jobs = scene_jobs_needing_normals(jobs, args)
    material_jobs = scene_jobs_needing_materials(jobs, args)
    if args.normal_preset is not None:
        run_geowizard_scene(scene_dir, normal_jobs, args)
    if args.material_preset is not None:
        run_rgbx_scene(scene_dir, material_jobs, args)


def process_job(job: PriorJob, args):
    if args.normal_command is not None and (not args.skip_existing or not job.output_normal_npy.exists()):
        run_command(args.normal_command, job, "normal", args.dry_run)
    if args.material_command is not None and (
        not args.skip_existing
        or not job.output_albedo_jpg.exists()
        or not job.output_roughness_jpg.exists()
    ):
        run_command(args.material_command, job, "material", args.dry_run)


def main():
    args = parse_args()
    validate_args(args)
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Processed data root not found: {data_root}")

    scene_dirs = list_scene_dirs(data_root, args.scene_ids)
    if not scene_dirs:
        raise ValueError(f"No scene directories found under {data_root}")

    jobs: List[PriorJob] = []
    for scene_dir in scene_dirs:
        jobs.extend(build_jobs(scene_dir))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    if not jobs:
        raise ValueError("No image jobs found")

    manifest_path = Path(args.manifest_path) if args.manifest_path else data_root / "prior_jobs.jsonl"
    write_manifest(manifest_path, jobs)
    print(f"Wrote job manifest: {manifest_path}", flush=True)
    print(f"Discovered {len(jobs)} image jobs across {len(scene_dirs)} scenes", flush=True)

    failures: List[str] = []
    if args.normal_preset is not None or args.material_preset is not None:
        scene_job_map = group_jobs_by_scene(jobs)
        scene_items = [(data_root / scene, scene_job_map[scene]) for scene in sorted(scene_job_map)]
        print(f"Running preset extraction for {len(scene_items)} scenes", flush=True)
        if args.workers <= 1:
            for scene_dir, scene_jobs in scene_items:
                try:
                    process_scene(scene_dir, scene_jobs, args)
                except Exception as exc:
                    failures.append(str(exc))
                    print(str(exc))
                    if args.fail_fast:
                        break
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(process_scene, scene_dir, scene_jobs, args): scene_dir
                    for scene_dir, scene_jobs in scene_items
                }
                for future in as_completed(future_map):
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(str(exc))
                        print(str(exc))
                        if args.fail_fast:
                            break
    else:
        filtered_jobs = [job for job in jobs if not should_skip(job, args)]
        print(f"Running {len(filtered_jobs)} jobs after skip filtering", flush=True)
        if not filtered_jobs:
            return
        if args.workers <= 1:
            for job in filtered_jobs:
                try:
                    process_job(job, args)
                except Exception as exc:
                    failures.append(str(exc))
                    print(str(exc))
                    if args.fail_fast:
                        break
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_map = {executor.submit(process_job, job, args): job for job in filtered_jobs}
                for future in as_completed(future_map):
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(str(exc))
                        print(str(exc))
                        if args.fail_fast:
                            break

    if failures:
        raise SystemExit(f"Prior extraction finished with {len(failures)} failure(s)")

    print("Prior extraction completed successfully", flush=True)


if __name__ == "__main__":
    main()