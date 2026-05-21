import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm
import cv2
import torch


LIDAR_BIN_COLUMNS = 14
POINT_SLICE = slice(3, 6)
GROUND_LABEL_INDEX = 10
INTENSITY_INDEX = 11
LASER_ID_INDEX = 13
NUM_BEAMS = 64
CAM_IDS = [0, 1, 2, 3, 4]
WAYMO_IMAGE_SIZES = {
    0: (1280, 1920),
    1: (1280, 1920),
    2: (1280, 1920),
    3: (866, 1920),
    4: (866, 1920),
}
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
TOP_LIDAR_VERTICAL_ANGLE_MIN = -0.31
TOP_LIDAR_VERTICAL_ANGLE_MAX = 0.04


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and optionally apply a beam-conditioned LUT for Waymo LiDAR intensity normalization"
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
        help="Optional scene ids to process, e.g. 023 114 327",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=0.25,
        help="XY grid size in meters used for the cell-based LUT statistics",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=256,
        help="Number of intensity bins in the LUT",
    )
    parser.add_argument(
        "--quantile_low",
        type=float,
        default=0.5,
        help="Lower percentile used to clip road intensity values before binning",
    )
    parser.add_argument(
        "--quantile_high",
        type=float,
        default=99.5,
        help="Upper percentile used to clip road intensity values before binning",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=500000,
        help="Maximum number of road-point intensity samples used to estimate bin edges",
    )
    parser.add_argument(
        "--min_cell_points",
        type=int,
        default=8,
        help="Minimum number of road points required for a cell to contribute to the LUT",
    )
    parser.add_argument(
        "--min_other_points",
        type=int,
        default=3,
        help="Minimum number of points from other beams required when computing a cell mean",
    )
    parser.add_argument(
        "--only_ground",
        action="store_true",
        help="Use only ground-labeled points for LUT estimation",
    )
    parser.add_argument(
        "--top_lidar_only",
        action="store_true",
        help="Restrict LUT estimation to the top LiDAR beam group (laser id 0)",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=None,
        help="Optional Euclidean range cutoff in meters for points used in LUT estimation",
    )
    parser.add_argument(
        "--min_range",
        type=float,
        default=None,
        help="Optional minimum Euclidean range in meters for points used in LUT estimation",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the LUT .npz file; defaults to <data_root>/waymo_intensity_lut.npz",
    )
    parser.add_argument(
        "--normalized_dirname",
        default=None,
        help="Optional subdirectory name for writing normalized lidar scan .bin files inside each scene. Do not use 'intensity': training expects projected camera-space .npy maps there, not scan bins.",
    )
    parser.add_argument(
        "--overwrite_normalized",
        action="store_true",
        help="Overwrite normalized lidar bins if they already exist",
    )
    parser.add_argument(
        "--project_to_intensity",
        action="store_true",
        help="After writing normalized lidar bins, project them into camera-space intensity/{frame}_{cam}.npy maps for training supervision.",
    )
    parser.add_argument(
        "--projected_dirname",
        default="intensity",
        help="Subdirectory name for projected camera-space intensity maps.",
    )
    parser.add_argument(
        "--overwrite_projected",
        action="store_true",
        help="Overwrite projected camera-space intensity maps if they already exist.",
    )
    parser.add_argument(
        "--projection_window",
        type=int,
        default=0,
        help="Number of temporally local LiDAR scans to project per frame; 0 keeps the old all-scans projection.",
    )
    parser.add_argument(
        "--projection_dilation_kernel",
        type=int,
        default=1,
        help="Dilation kernel for projected intensity values. Use 1 for sparse training targets; 5 approximates the old visualization-style target.",
    )
    parser.add_argument(
        "--projection_normalize",
        choices=["fixed", "frame_percentile", "none"],
        default="fixed",
        help="How to normalize projected intensity maps. 'fixed' uses a scene-stable clip/max scale; avoid 'frame_percentile' for training.",
    )
    parser.add_argument(
        "--projection_clip_max",
        type=float,
        default=2.0,
        help="Maximum projected intensity value before fixed normalization.",
    )
    parser.add_argument(
        "--projection_percentile_low",
        type=float,
        default=2.0,
        help="Lower percentile for --projection_normalize frame_percentile.",
    )
    parser.add_argument(
        "--projection_percentile_high",
        type=float,
        default=98.0,
        help="Upper percentile for --projection_normalize frame_percentile.",
    )
    parser.add_argument(
        "--projection_gamma",
        type=float,
        default=1.0,
        help="Gamma applied after projection normalization. Use 1.0 for training targets.",
    )
    parser.add_argument(
        "--limit_scans",
        type=int,
        default=0,
        help="Limit the number of scans processed for testing; 0 means no limit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for intensity sampling",
    )
    parser.add_argument(
        "--per_scene_lut",
        action="store_true",
        help="Build one LUT per scene instead of a single dataset-level LUT",
    )
    parser.add_argument(
        "--per_beam_edges",
        action="store_true",
        help="Estimate separate intensity bin edges for each beam",
    )
    parser.add_argument(
        "--target_stat",
        choices=["mean", "median"],
        default="median",
        help="Statistic used to estimate the target intensity from other beams inside each road cell",
    )
    parser.add_argument(
        "--beam_bias_correction",
        choices=["none", "mean", "median"],
        default="none",
        help="Optional post-fit per-beam correction applied to the LUT rows",
    )
    return parser.parse_args()


def list_scene_dirs(data_root: Path, scene_ids: List[str]) -> List[Path]:
    if scene_ids is None:
        return sorted(path for path in data_root.iterdir() if path.is_dir())
    wanted = {str(scene).zfill(3) for scene in scene_ids}
    return [data_root / scene for scene in sorted(wanted) if (data_root / scene).is_dir()]


def iter_lidar_files(scene_dirs: Iterable[Path]) -> List[Tuple[str, Path]]:
    scan_files: List[Tuple[str, Path]] = []
    for scene_dir in scene_dirs:
        lidar_dir = scene_dir / "lidar"
        if not lidar_dir.exists():
            continue
        for scan_path in sorted(lidar_dir.glob("*.bin")):
            scan_files.append((scene_dir.name, scan_path))
    return scan_files


def load_lidar_info(scan_path: Path) -> np.ndarray:
    lidar_info = np.memmap(scan_path, dtype=np.float32, mode="r")
    if lidar_info.size % LIDAR_BIN_COLUMNS != 0:
        raise ValueError(f"Unexpected lidar shape in {scan_path}: {lidar_info.size} values")
    return lidar_info.reshape(-1, LIDAR_BIN_COLUMNS)


def select_points(lidar_info: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(lidar_info[:, POINT_SLICE], dtype=np.float32)
    intensities = np.asarray(lidar_info[:, INTENSITY_INDEX], dtype=np.float32)
    sensor_ids = np.asarray(lidar_info[:, LASER_ID_INDEX], dtype=np.int32)

    valid_mask = np.isfinite(intensities) & np.all(np.isfinite(points), axis=1)
    # Only calibrate the Top LiDAR (sensor_id 0) to remove rings
    valid_mask &= (sensor_ids == 0)

    ranges = np.linalg.norm(points, axis=1)
    if args.min_range is not None:
        valid_mask &= ranges >= args.min_range
    if args.max_range is not None:
        valid_mask &= ranges <= args.max_range

    if args.only_ground:
        valid_mask &= np.asarray(lidar_info[:, GROUND_LABEL_INDEX] > 0.5)

    points = points[valid_mask]
    intensities = intensities[valid_mask]
    if points.shape[0] == 0:
        return points, intensities, np.empty(0, dtype=np.int32)

    return points, intensities, compute_top_lidar_beam_ids(points)


def compute_top_lidar_beam_ids(points: np.ndarray) -> np.ndarray:
    dist_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    angles = np.arctan2(points[:, 2], dist_xy)
    beam_ids = (
        (angles - TOP_LIDAR_VERTICAL_ANGLE_MIN)
        / (TOP_LIDAR_VERTICAL_ANGLE_MAX - TOP_LIDAR_VERTICAL_ANGLE_MIN + 1e-6)
        * (NUM_BEAMS - 1)
    )
    return np.clip(beam_ids, 0, NUM_BEAMS - 1).astype(np.int32)


def sample_intensities(scan_files: List[Tuple[str, Path]], args) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    samples: List[np.ndarray] = []
    beam_samples: Dict[int, List[np.ndarray]] = {beam_id: [] for beam_id in range(NUM_BEAMS)}
    remaining = args.sample_limit
    progress = tqdm(scan_files, desc="Sampling intensities", dynamic_ncols=True)
    for _, scan_path in progress:
        if remaining <= 0:
            break
        lidar_info = load_lidar_info(scan_path)
        _, intensities, laser_ids = select_points(lidar_info, args)
        if intensities.size == 0:
            continue
        take = min(remaining, min(4096, intensities.size))
        if take < intensities.size:
            choice = rng.choice(intensities.size, size=take, replace=False)
            intensities = intensities[choice]
            laser_ids = laser_ids[choice]
        samples.append(intensities.astype(np.float32))
        clipped_beams = np.clip(laser_ids, 0, NUM_BEAMS - 1)
        for beam_id in range(NUM_BEAMS):
            beam_mask = clipped_beams == beam_id
            if beam_mask.any():
                beam_samples[beam_id].append(intensities[beam_mask].astype(np.float32))
        remaining -= take
        progress.set_postfix(remaining=remaining)
    if not samples:
        raise ValueError("No road-point intensities found for LUT estimation")
    merged_beam_samples = {
        beam_id: np.concatenate(values, axis=0) if values else np.empty(0, dtype=np.float32)
        for beam_id, values in beam_samples.items()
    }
    return np.concatenate(samples, axis=0), merged_beam_samples


def build_edge_row(samples: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray]:
    low = np.percentile(samples, args.quantile_low)
    high = np.percentile(samples, args.quantile_high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Non-finite intensity quantiles encountered")
    if high <= low:
        high = low + 1e-6
    edges = np.linspace(low, high, args.num_bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def build_edges(samples: np.ndarray, beam_samples: Dict[int, np.ndarray], args) -> Tuple[np.ndarray, np.ndarray]:
    global_edges, global_centers = build_edge_row(samples, args)
    if not args.per_beam_edges:
        return global_edges, global_centers

    beam_edges = np.zeros((NUM_BEAMS, args.num_bins + 1), dtype=np.float32)
    beam_centers = np.zeros((NUM_BEAMS, args.num_bins), dtype=np.float32)
    for beam_id in range(NUM_BEAMS):
        beam_values = beam_samples[beam_id]
        if beam_values.size == 0:
            beam_edges[beam_id] = global_edges
            beam_centers[beam_id] = global_centers
            continue
        row_edges, row_centers = build_edge_row(beam_values, args)
        beam_edges[beam_id] = row_edges
        beam_centers[beam_id] = row_centers
    return beam_edges, beam_centers


def digitize_intensity(intensities: np.ndarray, edges: np.ndarray, laser_ids: np.ndarray = None) -> np.ndarray:
    if edges.ndim == 1:
        clipped = np.clip(intensities, edges[0], edges[-1])
        bins = np.searchsorted(edges, clipped, side="right") - 1
        return np.clip(bins, 0, len(edges) - 2).astype(np.int32)

    if laser_ids is None:
        raise ValueError("laser_ids are required when using per-beam edges")

    clipped_beams = np.clip(laser_ids, 0, edges.shape[0] - 1)
    bins = np.empty(intensities.shape[0], dtype=np.int32)
    for beam_id in range(edges.shape[0]):
        mask = clipped_beams == beam_id
        if not mask.any():
            continue
        row_edges = edges[beam_id]
        clipped = np.clip(intensities[mask], row_edges[0], row_edges[-1])
        row_bins = np.searchsorted(row_edges, clipped, side="right") - 1
        bins[mask] = np.clip(row_bins, 0, len(row_edges) - 2)
    return bins.astype(np.int32)


def clip_intensity_to_edges(intensities: np.ndarray, edges: np.ndarray, laser_ids: np.ndarray = None) -> np.ndarray:
    if edges.ndim == 1:
        return np.clip(intensities, edges[0], edges[-1]).astype(np.float32)

    if laser_ids is None:
        raise ValueError("laser_ids are required when using per-beam edges")

    clipped_beams = np.clip(laser_ids, 0, edges.shape[0] - 1)
    clipped = np.empty_like(intensities, dtype=np.float32)
    for beam_id in range(edges.shape[0]):
        mask = clipped_beams == beam_id
        if not mask.any():
            continue
        row_edges = edges[beam_id]
        clipped[mask] = np.clip(intensities[mask], row_edges[0], row_edges[-1])
    return clipped


def make_cell_keys(points: np.ndarray, cell_size: float) -> np.ndarray:
    cell_x = np.floor(points[:, 0] / cell_size).astype(np.int32)
    cell_y = np.floor(points[:, 1] / cell_size).astype(np.int32)
    keys = np.empty(points.shape[0], dtype=[("x", np.int32), ("y", np.int32)])
    keys["x"] = cell_x
    keys["y"] = cell_y
    return keys


def accumulate_lut(
    scan_files: List[Tuple[str, Path]],
    edges: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    lut_sums = np.zeros((NUM_BEAMS, args.num_bins), dtype=np.float64)
    lut_counts = np.zeros((NUM_BEAMS, args.num_bins), dtype=np.int64)
    if edges.ndim == 1:
        global_bin_sums = np.zeros(args.num_bins, dtype=np.float64)
        global_bin_counts = np.zeros(args.num_bins, dtype=np.int64)
    else:
        global_bin_sums = np.zeros((NUM_BEAMS, args.num_bins), dtype=np.float64)
        global_bin_counts = np.zeros((NUM_BEAMS, args.num_bins), dtype=np.int64)
    stats = {
        "total_scans": 0,
        "used_scans": 0,
        "total_points": 0,
        "used_points": 0,
        "used_cells": 0,
    }

    progress = tqdm(scan_files, desc="Accumulating LUT", dynamic_ncols=True)
    for _, scan_path in progress:
        stats["total_scans"] += 1
        lidar_info = load_lidar_info(scan_path)
        stats["total_points"] += int(lidar_info.shape[0])
        points, intensities, laser_ids = select_points(lidar_info, args)
        if intensities.size == 0:
            progress.set_postfix(used_scans=stats["used_scans"], used_cells=stats["used_cells"])
            continue

        bins = digitize_intensity(intensities, edges, laser_ids)
        clipped_intensities = clip_intensity_to_edges(intensities, edges, laser_ids)
        if edges.ndim == 1:
            np.add.at(global_bin_sums, bins, clipped_intensities)
            np.add.at(global_bin_counts, bins, 1)
        else:
            clipped_beams = np.clip(laser_ids, 0, NUM_BEAMS - 1)
            np.add.at(global_bin_sums, (clipped_beams, bins), clipped_intensities)
            np.add.at(global_bin_counts, (clipped_beams, bins), 1)

        cell_keys = make_cell_keys(points, args.cell_size)
        _, inverse, counts = np.unique(cell_keys, return_inverse=True, return_counts=True)
        order = np.argsort(inverse)
        offsets = np.concatenate(([0], np.cumsum(counts)))

        scan_used = False
        for cell_index in range(len(counts)):
            start, end = offsets[cell_index], offsets[cell_index + 1]
            if end - start < args.min_cell_points:
                continue
            idx = order[start:end]
            cell_beams = laser_ids[idx]
            cell_intensities = clipped_intensities[idx]
            cell_bins = bins[idx]

            unique_beams, beam_inverse = np.unique(cell_beams, return_inverse=True)
            if unique_beams.size < 2 and not args.top_lidar_only:
                continue

            beam_counts = np.bincount(beam_inverse)
            beam_sums = None
            total_sum = None
            total_count = int(cell_intensities.size)
            if args.target_stat == "mean":
                beam_sums = np.bincount(beam_inverse, weights=cell_intensities)
                total_sum = float(cell_intensities.sum())
            beam_to_local = {int(beam): local_idx for local_idx, beam in enumerate(unique_beams.tolist())}

            unique_pairs = np.unique(np.stack([cell_beams, cell_bins], axis=1), axis=0)
            for beam_id, bin_id in unique_pairs:
                local_beam_idx = beam_to_local[int(beam_id)]
                other_count = total_count - int(beam_counts[local_beam_idx])
                if other_count < args.min_other_points:
                    continue
                if args.target_stat == "median":
                    other_values = cell_intensities[cell_beams != beam_id]
                    target_value = float(np.median(other_values))
                else:
                    target_value = (total_sum - float(beam_sums[local_beam_idx])) / other_count
                lut_sums[int(beam_id), int(bin_id)] += target_value
                lut_counts[int(beam_id), int(bin_id)] += 1
                scan_used = True

            stats["used_cells"] += 1

        if scan_used:
            stats["used_scans"] += 1
            stats["used_points"] += int(intensities.size)
        progress.set_postfix(used_scans=stats["used_scans"], used_cells=stats["used_cells"])

    return lut_sums, lut_counts, global_bin_sums, global_bin_counts, stats


def finalize_lut(
    lut_sums: np.ndarray,
    lut_counts: np.ndarray,
    global_bin_sums: np.ndarray,
    global_bin_counts: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    lut = np.full_like(lut_sums, np.nan, dtype=np.float32)
    valid = lut_counts > 0
    lut[valid] = (lut_sums[valid] / lut_counts[valid]).astype(np.float32)

    global_bin_mean = np.full_like(centers, np.nan, dtype=np.float32)
    valid_global = global_bin_counts > 0
    global_bin_mean[valid_global] = (global_bin_sums[valid_global] / global_bin_counts[valid_global]).astype(np.float32)
    global_fallback = np.where(np.isfinite(global_bin_mean), global_bin_mean, centers.astype(np.float32))

    if centers.ndim == 1:
        bin_positions = np.arange(len(centers), dtype=np.float32)
    else:
        bin_positions = np.arange(centers.shape[1], dtype=np.float32)
    for beam_id in range(lut.shape[0]):
        row = lut[beam_id]
        row_valid = np.isfinite(row)
        if row_valid.any():
            lut[beam_id] = np.interp(
                bin_positions,
                bin_positions[row_valid],
                row[row_valid],
            ).astype(np.float32)
        else:
            lut[beam_id] = global_fallback[beam_id] if global_fallback.ndim == 2 else global_fallback
        missing = ~np.isfinite(lut[beam_id])
        if missing.any():
            if global_fallback.ndim == 2:
                lut[beam_id, missing] = global_fallback[beam_id, missing]
            else:
                lut[beam_id, missing] = global_fallback[missing]

    return lut


def compute_beam_bias_factors(
    scan_files: List[Tuple[str, Path]],
    lut: np.ndarray,
    edges: np.ndarray,
    args,
) -> np.ndarray:
    if args.beam_bias_correction == "none":
        return np.ones(NUM_BEAMS, dtype=np.float32)

    values_per_beam: Dict[int, List[np.ndarray]] = {beam_id: [] for beam_id in range(NUM_BEAMS)}
    progress = tqdm(scan_files, desc="Estimating beam bias", dynamic_ncols=True)
    for _, scan_path in progress:
        lidar_info = load_lidar_info(scan_path)
        _, intensities, laser_ids = select_points(lidar_info, args)
        if intensities.size == 0:
            continue
        bins = digitize_intensity(intensities, edges, laser_ids)
        normalized = lut[np.clip(laser_ids, 0, lut.shape[0] - 1), bins]
        clipped_beams = np.clip(laser_ids, 0, NUM_BEAMS - 1)
        for beam_id in range(NUM_BEAMS):
            mask = clipped_beams == beam_id
            if mask.any():
                values_per_beam[beam_id].append(normalized[mask].astype(np.float32))

    beam_stats = np.full(NUM_BEAMS, np.nan, dtype=np.float32)
    for beam_id in range(NUM_BEAMS):
        if not values_per_beam[beam_id]:
            continue
        beam_values = np.concatenate(values_per_beam[beam_id], axis=0)
        if args.beam_bias_correction == "median":
            beam_stats[beam_id] = float(np.median(beam_values))
        else:
            beam_stats[beam_id] = float(np.mean(beam_values))

    valid = np.isfinite(beam_stats) & (beam_stats > 0)
    if not valid.any():
        return np.ones(NUM_BEAMS, dtype=np.float32)

    target = float(np.mean(beam_stats[valid]))
    factors = np.ones(NUM_BEAMS, dtype=np.float32)
    factors[valid] = target / beam_stats[valid]
    return factors


def apply_beam_bias_factors(lut: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return (lut * factors[:, None]).astype(np.float32)


def save_lut(
    output_path: Path,
    lut: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    stats: Dict[str, int],
    args,
    beam_bias_factors: np.ndarray,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cell_size": args.cell_size,
        "num_bins": args.num_bins,
        "quantile_low": args.quantile_low,
        "quantile_high": args.quantile_high,
        "only_ground": args.only_ground,
        "top_lidar_only": args.top_lidar_only,
        "min_range": args.min_range,
        "max_range": args.max_range,
        "per_scene_lut": args.per_scene_lut,
        "per_beam_edges": args.per_beam_edges,
        "target_stat": args.target_stat,
        "beam_bias_correction": args.beam_bias_correction,
        **stats,
    }
    np.savez_compressed(
        output_path,
        lut=lut,
        edges=edges,
        centers=centers,
        counts=counts,
        beam_bias_factors=beam_bias_factors.astype(np.float32),
        metadata_json=json.dumps(metadata, indent=2),
    )


def load_existing_lut(output_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(output_path, allow_pickle=False) as data:
        missing_keys = [key for key in ("lut", "edges") if key not in data]
        if missing_keys:
            raise KeyError(f"Existing LUT file {output_path} is missing keys: {missing_keys}")
        lut = data["lut"].astype(np.float32)
        edges = data["edges"].astype(np.float32)
    return lut, edges


def write_normalized_bins(scan_files: List[Tuple[str, Path]], lut: np.ndarray, edges: np.ndarray, dirname: str, overwrite: bool):
    progress = tqdm(scan_files, desc="Writing normalized bins", dynamic_ncols=True)
    for scene_name, scan_path in progress:
        scene_dir = scan_path.parent.parent
        output_dir = scene_dir / dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / scan_path.name
        
        if output_path.exists() and not overwrite: continue
            
        lidar_info = np.array(load_lidar_info(scan_path), copy=True)
        points = np.asarray(lidar_info[:, POINT_SLICE], dtype=np.float32)
        intensities = np.asarray(lidar_info[:, INTENSITY_INDEX], dtype=np.float32)
        sensor_ids = np.asarray(lidar_info[:, LASER_ID_INDEX], dtype=np.int32)
        
        # Identify beams for Top LiDAR points only
        top_mask = (sensor_ids == 0)
        if not top_mask.any():
            lidar_info[:, INTENSITY_INDEX] = 0
            lidar_info.astype(np.float32).tofile(output_path)
            continue

        beam_ids = compute_top_lidar_beam_ids(points[top_mask])

        # Apply LUT calibration
        bins = digitize_intensity(intensities[top_mask], edges, beam_ids)
        normalized = lut[beam_ids, bins]
        
        # Apply Distance Compensation (Reflectance calculation)
        dist = np.linalg.norm(points[top_mask], axis=1)
        # reflectance = normalized * ((dist / 20.0) ** 2)
        reflectance = normalized
        
        # Update only Top LiDAR points (set others to 0 or keep as is)
        lidar_info[~top_mask, INTENSITY_INDEX] = 0 
        lidar_info[top_mask, INTENSITY_INDEX] = reflectance.astype(np.float32)
        
        lidar_info.astype(np.float32).tofile(output_path)


def load_relative_ego_poses(scene_dir: Path, num_frames: int) -> List[np.ndarray]:
    ego_pose_dir = scene_dir / "ego_pose"
    ego_to_world_start = np.loadtxt(ego_pose_dir / "000.txt")
    ego_poses = []
    for frame_idx in range(num_frames):
        ego_to_world_current = np.loadtxt(ego_pose_dir / f"{frame_idx:03d}.txt")
        ego_poses.append(np.linalg.inv(ego_to_world_start) @ ego_to_world_current)
    return ego_poses


def load_camera_intrinsics(scene_dir: Path) -> Dict[int, np.ndarray]:
    intrinsics = {}
    for cam_id in CAM_IDS:
        intrinsic = np.loadtxt(scene_dir / "intrinsics" / f"{cam_id}.txt")
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsics[cam_id] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    return intrinsics


def load_camera_extrinsics(scene_dir: Path) -> Dict[int, np.ndarray]:
    cam_to_ego = {}
    for cam_id in CAM_IDS:
        cam_to_ego_raw = np.loadtxt(scene_dir / "extrinsics" / f"{cam_id}.txt")
        cam_to_ego[cam_id] = cam_to_ego_raw @ OPENCV2DATASET
    return cam_to_ego


def project_points(points_world: np.ndarray, cam_to_world: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(cam_to_world)
    cam_points = (world_to_cam[:3, :3] @ points_world.T + world_to_cam[:3, 3:4]).T
    depth = cam_points[:, 2]
    uv = cam_points[:, :2] / np.clip(depth[:, None], 1e-6, None)
    pixels = (intrinsic[:2, :2] @ uv.T).T + intrinsic[:2, 2]
    return np.concatenate([pixels, depth[:, None]], axis=1)


def project_normalized_bins_to_intensity_maps(
    scene_dir: Path,
    lidar_dirname: str,
    output_dirname: str,
    overwrite: bool,
    projection_window: int = 0,
    projection_dilation_kernel: int = 1,
    projection_normalize: str = "fixed",
    projection_clip_max: float = 2.0,
    projection_percentile_low: float = 2.0,
    projection_percentile_high: float = 98.0,
    projection_gamma: float = 1.0,
):
    if projection_clip_max <= 0:
        raise ValueError("--projection_clip_max must be positive")
    projection_dilation_kernel = max(int(projection_dilation_kernel), 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lidar_dir = scene_dir / lidar_dirname
    scan_paths = sorted(lidar_dir.glob("*.bin"))
    output_dir = scene_dir / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    ego_poses = load_relative_ego_poses(scene_dir, len(scan_paths))
    camera_intrinsics = load_camera_intrinsics(scene_dir)
    camera_extrinsics = load_camera_extrinsics(scene_dir)

    all_pts, all_ints = [], []
    if projection_window > 0:
        print(f"Loading local projection clouds on CPU; window={projection_window} scans")
    else:
        print(f"Moving Global Cloud to {device}...")
    for i, path in enumerate(scan_paths):
        lidar_info = np.asarray(load_lidar_info(path))
        pts_veh = lidar_info[:, POINT_SLICE]
        intensities = np.clip(lidar_info[:, INTENSITY_INDEX], 0, projection_clip_max)
        
        # Transform to World
        pts_world = (ego_poses[i][:3, :3] @ pts_veh.T + ego_poses[i][:3, 3:4]).T
        all_pts.append(torch.from_numpy(pts_world).float())
        all_ints.append(torch.from_numpy(intensities).float())

    if projection_window > 0:
        global_pts = None
        global_int = None
    else:
        # Massive tensors on GPU
        global_pts = torch.cat(all_pts, dim=0).to(device) # [N, 3]
        global_int = torch.cat(all_ints, dim=0).to(device) # [N]
    
    # Pre-calculate a mask for the global cloud to keep only points within a 100m cube 
    # (Optional: speeds up math if you have millions of distant points)
    
    for frame_idx in tqdm(range(len(scan_paths)), desc="GPU Projecting"):
        for cam_id in CAM_IDS:
            output_path = output_dir / f"{frame_idx:03d}_{cam_id}.npy"
            if output_path.exists() and not overwrite: continue

            h, w = WAYMO_IMAGE_SIZES[cam_id]
            cam_to_world = torch.from_numpy(ego_poses[frame_idx] @ camera_extrinsics[cam_id]).float().to(device)
            world_to_cam = torch.inverse(cam_to_world)
            K = torch.from_numpy(camera_intrinsics[cam_id]).float().to(device)

            if projection_window > 0:
                half_window = projection_window // 2
                start_idx = max(0, frame_idx - half_window)
                end_idx = min(len(scan_paths), start_idx + projection_window)
                start_idx = max(0, end_idx - projection_window)
                local_pts = torch.cat(all_pts[start_idx:end_idx], dim=0).to(device)
                local_int = torch.cat(all_ints[start_idx:end_idx], dim=0).to(device)
            else:
                local_pts = global_pts
                local_int = global_int

            # 1. GPU Projection (The Matrix Math)
            # cam_pts = R * world_pts + T
            cam_pts = (world_to_cam[:3, :3] @ local_pts.T + world_to_cam[:3, 3:4]).T
            depth = cam_pts[:, 2]
            
            # Frustum Culling (Only points in front of camera)
            front_mask = depth > 0.5
            if not front_mask.any(): continue
            
            # Project to Pixels
            pixel_pts = (K @ cam_pts[front_mask].T).T
            u = (pixel_pts[:, 0] / pixel_pts[:, 2]).long()
            v = (pixel_pts[:, 1] / pixel_pts[:, 2]).long()
            
            # 2. Vectorized Validity Check
            img_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not img_mask.any(): continue
            
            u, v, d = u[img_mask], v[img_mask], depth[front_mask][img_mask]
            intensities = local_int[front_mask][img_mask]

            # 3. GPU Z-Buffer (Instant Sorting)
            # Sort by depth descending so closest points are assigned last
            sort_idx = torch.argsort(d, descending=True)
            u_final, v_vinal, i_final = u[sort_idx], v[sort_idx], intensities[sort_idx]

            # 4. Create Image
            intensity_map = torch.zeros((h, w), device=device)
            intensity_map[v_vinal, u_final] = i_final
            
            # Back to NumPy for OpenCV and Saving
            img_np = intensity_map.cpu().numpy()

            if projection_dilation_kernel > 1:
                kernel = np.ones((projection_dilation_kernel, projection_dilation_kernel), np.uint8)
                img_np = cv2.dilate(img_np, kernel)

            if projection_normalize == "fixed":
                img_np = np.clip(img_np, 0.0, projection_clip_max) / max(projection_clip_max, 1e-6)
            elif projection_normalize == "frame_percentile" and img_np.max() > 0:
                valid = img_np[img_np > 0]
                v_min = np.percentile(valid, projection_percentile_low)
                v_max = np.percentile(valid, projection_percentile_high)
                img_np = np.clip((img_np - v_min) / (v_max - v_min + 1e-6), 0, 1)
            elif projection_normalize == "none":
                img_np = np.maximum(img_np, 0.0)

            if projection_gamma != 1.0 and img_np.max() > 0:
                img_np = np.power(np.clip(img_np, 0.0, None), projection_gamma)

            np.save(output_path, img_np[..., None].astype(np.float32))


def format_edge_summary(edges: np.ndarray) -> str:
    if edges.ndim == 1:
        return f"Intensity binning range: [{edges[0]:.6f}, {edges[-1]:.6f}] with {len(edges) - 1} bins"
    ranges = ", ".join(
        f"beam {beam_id}: [{row[0]:.6f}, {row[-1]:.6f}]"
        for beam_id, row in enumerate(edges)
    )
    return f"Per-beam intensity binning with {edges.shape[1] - 1} bins each ({ranges})"


def build_for_scan_files(scan_files: List[Tuple[str, Path]], args, output_path: Path, scope_label: str):
    print(f"Found {len(scan_files)} lidar scans for {scope_label}")
    if output_path.exists():
        print(f"Reusing existing LUT at {output_path}; skipping sampling and accumulation")
        lut, edges = load_existing_lut(output_path)
        print(format_edge_summary(edges))
    else:
        samples, beam_samples = sample_intensities(scan_files, args)
        edges, centers = build_edges(samples, beam_samples, args)
        print(format_edge_summary(edges))

        lut_sums, lut_counts, global_bin_sums, global_bin_counts, stats = accumulate_lut(scan_files, edges, args)
        lut = finalize_lut(lut_sums, lut_counts, global_bin_sums, global_bin_counts, centers)
        beam_bias_factors = compute_beam_bias_factors(scan_files, lut, edges, args)
        if args.beam_bias_correction != "none":
            lut = apply_beam_bias_factors(lut, beam_bias_factors)
            print(f"Applied beam bias factors: {beam_bias_factors.tolist()}")
        save_lut(output_path, lut, edges, centers, lut_counts, stats, args, beam_bias_factors)
        print(f"Saved LUT to {output_path}")
        print(json.dumps(stats, indent=2))

    if args.normalized_dirname:
        write_normalized_bins(
            scan_files=scan_files,
            lut=lut,
            edges=edges,
            dirname=args.normalized_dirname,
            overwrite=args.overwrite_normalized,
        )
        if args.project_to_intensity:
            scene_dirs = sorted({scan_path.parent.parent for _, scan_path in scan_files})
            for scene_dir in scene_dirs:
                project_normalized_bins_to_intensity_maps(
                    scene_dir=scene_dir,
                    lidar_dirname=args.normalized_dirname,
                    output_dirname=args.projected_dirname,
                    overwrite=args.overwrite_projected,
                    projection_window=args.projection_window,
                    projection_dilation_kernel=args.projection_dilation_kernel,
                    projection_normalize=args.projection_normalize,
                    projection_clip_max=args.projection_clip_max,
                    projection_percentile_low=args.projection_percentile_low,
                    projection_percentile_high=args.projection_percentile_high,
                    projection_gamma=args.projection_gamma,
                )
    elif args.project_to_intensity:
        raise ValueError("--project_to_intensity requires --normalized_dirname so the script knows which normalized lidar scans to project.")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Processed data root not found: {data_root}")

    scene_dirs = list_scene_dirs(data_root, args.scene_ids)
    if not scene_dirs:
        raise ValueError(f"No scene directories found under {data_root}")

    scan_files = iter_lidar_files(scene_dirs)
    if args.limit_scans > 0:
        scan_files = scan_files[: args.limit_scans]
    if not scan_files:
        raise ValueError("No lidar scans found for LUT estimation")

    if args.per_scene_lut:
        for scene_dir in scene_dirs:
            scene_scan_files = [(scene_name, scan_path) for scene_name, scan_path in scan_files if scene_name == scene_dir.name]
            if not scene_scan_files:
                continue
            if args.limit_scans > 0:
                scene_scan_files = scene_scan_files[: args.limit_scans]
            output_path = Path(args.output_path) if args.output_path else scene_dir / "waymo_intensity_lut.npz"
            build_for_scan_files(scene_scan_files, args, output_path, f"scene {scene_dir.name}")
    else:
        output_path = Path(args.output_path) if args.output_path else data_root / "waymo_intensity_lut.npz"
        build_for_scan_files(scan_files, args, output_path, f"{len(scene_dirs)} scenes")


if __name__ == "__main__":
    main()
