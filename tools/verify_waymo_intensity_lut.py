import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


LIDAR_BIN_COLUMNS = 14
POINT_SLICE = slice(3, 6)
GROUND_LABEL_INDEX = 10
INTENSITY_INDEX = 11
LASER_ID_INDEX = 13
NUM_BEAMS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify whether a Waymo LUT improved LiDAR intensity consistency"
    )
    parser.add_argument(
        "--data_root",
        default="data/waymo/processed/training",
        help="Root directory containing processed Waymo scene folders",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help="Optional scene ids to verify, e.g. 023 114 327",
    )
    parser.add_argument(
        "--lut_path",
        default=None,
        help="Path to the LUT .npz file; defaults to <data_root>/waymo_intensity_lut.npz",
    )
    parser.add_argument(
        "--normalized_dirname",
        default="lidar_normalized",
        help="Scene subdirectory containing normalized lidar bins",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=0.25,
        help="XY cell size in meters for the cross-beam consistency check",
    )
    parser.add_argument(
        "--only_ground",
        action="store_true",
        help="Restrict verification to ground-labeled points",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=None,
        help="Optional Euclidean range cutoff in meters",
    )
    parser.add_argument(
        "--limit_scans",
        type=int,
        default=0,
        help="Limit the number of scans for a quick verification run; 0 means no limit",
    )
    parser.add_argument(
        "--output_dir",
        default="debug/lut_verify",
        help="Directory to save verification figures and JSON summary",
    )
    parser.add_argument(
        "--sample_per_beam",
        type=int,
        default=50000,
        help="Maximum number of points per beam used in histogram plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for plot subsampling",
    )
    return parser.parse_args()


def list_scene_dirs(data_root: Path, scene_ids: List[str]) -> List[Path]:
    if scene_ids is None:
        return sorted(path for path in data_root.iterdir() if path.is_dir())
    wanted = {str(scene).zfill(3) for scene in scene_ids}
    return [data_root / scene for scene in sorted(wanted) if (data_root / scene).is_dir()]


def iter_scan_pairs(scene_dirs: Iterable[Path], normalized_dirname: str) -> List[Tuple[str, Path, Path]]:
    pairs = []
    for scene_dir in scene_dirs:
        raw_dir = scene_dir / "lidar"
        normalized_dir = scene_dir / normalized_dirname
        if not raw_dir.exists() or not normalized_dir.exists():
            continue
        for raw_path in sorted(raw_dir.glob("*.bin")):
            normalized_path = normalized_dir / raw_path.name
            if normalized_path.exists():
                pairs.append((scene_dir.name, raw_path, normalized_path))
    return pairs


def load_lidar_info(scan_path: Path) -> np.ndarray:
    data = np.memmap(scan_path, dtype=np.float32, mode="r")
    if data.size % LIDAR_BIN_COLUMNS != 0:
        raise ValueError(f"Unexpected lidar shape in {scan_path}: {data.size} values")
    return data.reshape(-1, LIDAR_BIN_COLUMNS)


def select_points(lidar_info: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(lidar_info[:, POINT_SLICE], dtype=np.float32)
    intensities = np.asarray(lidar_info[:, INTENSITY_INDEX], dtype=np.float32)
    laser_ids = np.asarray(lidar_info[:, LASER_ID_INDEX], dtype=np.int32)

    valid = np.isfinite(intensities)
    valid &= np.all(np.isfinite(points), axis=1)
    if args.only_ground:
        valid &= np.asarray(lidar_info[:, GROUND_LABEL_INDEX] > 0.5)
    if args.max_range is not None:
        valid &= np.linalg.norm(points, axis=1) <= args.max_range
    return points[valid], intensities[valid], laser_ids[valid]


def make_cell_keys(points: np.ndarray, cell_size: float) -> np.ndarray:
    cell_x = np.floor(points[:, 0] / cell_size).astype(np.int32)
    cell_y = np.floor(points[:, 1] / cell_size).astype(np.int32)
    keys = np.empty(points.shape[0], dtype=[("x", np.int32), ("y", np.int32)])
    keys["x"] = cell_x
    keys["y"] = cell_y
    return keys


def beam_statistics(intensities: np.ndarray, laser_ids: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for beam_id in range(NUM_BEAMS):
        mask = laser_ids == beam_id
        if not mask.any():
            continue
        values = intensities[mask]
        stats[str(beam_id)] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
        }
    return stats


def collect_plot_samples(
    values: np.ndarray,
    beam_ids: np.ndarray,
    sample_per_beam: int,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    result: Dict[int, np.ndarray] = {}
    for beam_id in range(NUM_BEAMS):
        beam_values = values[beam_ids == beam_id]
        if beam_values.size == 0:
            continue
        if beam_values.size > sample_per_beam:
            idx = rng.choice(beam_values.size, size=sample_per_beam, replace=False)
            beam_values = beam_values[idx]
        result[beam_id] = beam_values
    return result


def cell_disagreement(points: np.ndarray, values: np.ndarray, beam_ids: np.ndarray, cell_size: float) -> Tuple[float, int]:
    if values.size == 0:
        return float("nan"), 0
    keys = make_cell_keys(points, cell_size)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    disagreements = []
    for cell_index in range(len(counts)):
        start, end = offsets[cell_index], offsets[cell_index + 1]
        idx = order[start:end]
        cell_beams = beam_ids[idx]
        cell_values = values[idx]
        unique_beams = np.unique(cell_beams)
        if unique_beams.size < 2:
            continue
        beam_means = []
        for beam_id in unique_beams:
            beam_means.append(cell_values[cell_beams == beam_id].mean())
        disagreements.append(float(np.std(beam_means)))

    if not disagreements:
        return float("nan"), 0
    return float(np.mean(disagreements)), len(disagreements)


def load_lut_summary(lut_path: Path) -> Dict[str, object]:
    lut_file = np.load(lut_path, allow_pickle=False)
    metadata_json = str(lut_file["metadata_json"])
    metadata = json.loads(metadata_json)
    return {
        "lut_shape": list(lut_file["lut"].shape),
        "edges_shape": list(lut_file["edges"].shape),
        "counts_nonzero": int((lut_file["counts"] > 0).sum()),
        "beam_bias_factors": lut_file["beam_bias_factors"].astype(float).tolist() if "beam_bias_factors" in lut_file else None,
        "metadata": metadata,
    }


def make_verdict(raw_beam_mean_std: float, normalized_beam_mean_std: float, raw_cell: float, normalized_cell: float) -> Tuple[str, Dict[str, float]]:
    deltas = {
        "beam_mean_std_delta": float(normalized_beam_mean_std - raw_beam_mean_std),
        "cell_disagreement_delta": float(normalized_cell - raw_cell),
    }
    beam_improved = normalized_beam_mean_std < raw_beam_mean_std
    cell_improved = normalized_cell < raw_cell
    if beam_improved and cell_improved:
        return "improved", deltas
    if cell_improved and not beam_improved:
        return "mixed", deltas
    if beam_improved and not cell_improved:
        return "mixed", deltas
    return "worse", deltas


def plot_histograms(raw_samples: Dict[int, np.ndarray], normalized_samples: Dict[int, np.ndarray], output_dir: Path):
    fig = plt.figure(figsize=(14, 8))
    for beam_id, values in raw_samples.items():
        plt.hist(values, bins=80, alpha=0.35, density=True, label=f"raw beam {beam_id}")
    for beam_id, values in normalized_samples.items():
        plt.hist(values, bins=80, alpha=0.35, density=True, histtype="step", linewidth=1.5, label=f"norm beam {beam_id}")
    plt.title("Per-beam intensity distributions before and after normalization")
    plt.xlabel("Intensity")
    plt.ylabel("Density")
    plt.legend(ncol=2, fontsize=8)
    fig.savefig(output_dir / "beam_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beam_means(raw_stats: Dict[str, Dict[str, float]], normalized_stats: Dict[str, Dict[str, float]], output_dir: Path):
    beams = sorted(set(raw_stats) | set(normalized_stats), key=int)
    raw_means = [raw_stats.get(beam, {}).get("mean", np.nan) for beam in beams]
    norm_means = [normalized_stats.get(beam, {}).get("mean", np.nan) for beam in beams]
    x = np.arange(len(beams))
    width = 0.35

    fig = plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, raw_means, width=width, label="raw")
    plt.bar(x + width / 2, norm_means, width=width, label="normalized")
    plt.xticks(x, beams)
    plt.xlabel("Beam ID")
    plt.ylabel("Mean intensity")
    plt.title("Per-beam mean intensity")
    plt.legend()
    fig.savefig(output_dir / "beam_means.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Processed data root not found: {data_root}")

    lut_path = Path(args.lut_path) if args.lut_path else data_root / "waymo_intensity_lut.npz"
    lut_summary = None
    if lut_path.exists():
        lut_summary = load_lut_summary(lut_path)

    scene_dirs = list_scene_dirs(data_root, args.scene_ids)
    scan_pairs = iter_scan_pairs(scene_dirs, args.normalized_dirname)
    if args.limit_scans > 0:
        scan_pairs = scan_pairs[: args.limit_scans]
    if not scan_pairs:
        raise ValueError("No raw/normalized scan pairs found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    raw_all = []
    normalized_all = []
    beam_all = []
    sample_raw = {beam_id: [] for beam_id in range(NUM_BEAMS)}
    sample_norm = {beam_id: [] for beam_id in range(NUM_BEAMS)}
    raw_disagreements = []
    norm_disagreements = []
    raw_cells = 0
    norm_cells = 0

    progress = tqdm(scan_pairs, desc="Verifying LUT", dynamic_ncols=True)
    for _, raw_path, normalized_path in progress:
        raw_info = load_lidar_info(raw_path)
        norm_info = load_lidar_info(normalized_path)
        if raw_info.shape != norm_info.shape:
            raise ValueError(f"Shape mismatch between {raw_path} and {normalized_path}")

        raw_points, raw_intensity, raw_beams = select_points(raw_info, args)
        norm_points, norm_intensity, norm_beams = select_points(norm_info, args)
        if raw_points.shape != norm_points.shape or not np.array_equal(raw_beams, norm_beams):
            raise ValueError(f"Filtered raw/normalized point mismatch for {raw_path}")

        raw_all.append(raw_intensity)
        normalized_all.append(norm_intensity)
        beam_all.append(raw_beams)

        per_beam_raw = collect_plot_samples(raw_intensity, raw_beams, args.sample_per_beam, rng)
        per_beam_norm = collect_plot_samples(norm_intensity, norm_beams, args.sample_per_beam, rng)
        for beam_id, values in per_beam_raw.items():
            sample_raw[beam_id].append(values)
        for beam_id, values in per_beam_norm.items():
            sample_norm[beam_id].append(values)

        raw_disagreement, raw_count = cell_disagreement(raw_points, raw_intensity, raw_beams, args.cell_size)
        norm_disagreement, norm_count = cell_disagreement(norm_points, norm_intensity, norm_beams, args.cell_size)
        if np.isfinite(raw_disagreement):
            raw_disagreements.append(raw_disagreement)
            raw_cells += raw_count
        if np.isfinite(norm_disagreement):
            norm_disagreements.append(norm_disagreement)
            norm_cells += norm_count

        progress.set_postfix(raw_cell_std=np.mean(raw_disagreements) if raw_disagreements else np.nan,
                             norm_cell_std=np.mean(norm_disagreements) if norm_disagreements else np.nan)

    raw_values = np.concatenate(raw_all, axis=0)
    normalized_values = np.concatenate(normalized_all, axis=0)
    beam_ids = np.concatenate(beam_all, axis=0)

    raw_stats = beam_statistics(raw_values, beam_ids)
    normalized_stats = beam_statistics(normalized_values, beam_ids)

    raw_means = np.array([stats["mean"] for stats in raw_stats.values()], dtype=np.float32)
    norm_means = np.array([stats["mean"] for stats in normalized_stats.values()], dtype=np.float32)
    summary = {
        "lut": lut_summary,
        "num_scans": len(scan_pairs),
        "num_points": int(raw_values.size),
        "raw_beam_stats": raw_stats,
        "normalized_beam_stats": normalized_stats,
        "raw_beam_mean_std": float(np.std(raw_means)) if raw_means.size > 0 else float("nan"),
        "normalized_beam_mean_std": float(np.std(norm_means)) if norm_means.size > 0 else float("nan"),
        "raw_mean_cell_disagreement": float(np.mean(raw_disagreements)) if raw_disagreements else float("nan"),
        "normalized_mean_cell_disagreement": float(np.mean(norm_disagreements)) if norm_disagreements else float("nan"),
        "raw_cells_used": raw_cells,
        "normalized_cells_used": norm_cells,
    }
    verdict, deltas = make_verdict(
        summary["raw_beam_mean_std"],
        summary["normalized_beam_mean_std"],
        summary["raw_mean_cell_disagreement"],
        summary["normalized_mean_cell_disagreement"],
    )
    summary["deltas"] = deltas
    summary["verdict"] = verdict

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    raw_plot_samples = {
        beam_id: np.concatenate(values, axis=0)[: args.sample_per_beam]
        for beam_id, values in sample_raw.items()
        if values
    }
    norm_plot_samples = {
        beam_id: np.concatenate(values, axis=0)[: args.sample_per_beam]
        for beam_id, values in sample_norm.items()
        if values
    }
    plot_histograms(raw_plot_samples, norm_plot_samples, output_dir)
    plot_beam_means(raw_stats, normalized_stats, output_dir)

    print(f"Verdict: {verdict}")
    print(
        "Deltas: "
        f"beam_mean_std={deltas['beam_mean_std_delta']:+.6f}, "
        f"cell_disagreement={deltas['cell_disagreement_delta']:+.6f}"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved verification outputs to: {output_dir}")


if __name__ == "__main__":
    main()