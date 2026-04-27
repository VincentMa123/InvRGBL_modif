import argparse
import glob
import logging
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger()

CAMERA_NAMES = {
    0: "FRONT",
    1: "FRONT_LEFT",
    2: "FRONT_RIGHT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT",
}


def make_video_for_camera(image_dir: str, cam_id: int, out_path: str, fps: int):
    """Stitch all frames of a single camera into an MP4 video."""
    pattern = os.path.join(image_dir, f"*_{cam_id}.jpg")
    frames = sorted(glob.glob(pattern))
    if len(frames) == 0:
        logger.warning(f"No frames found for camera {cam_id} in {image_dir}")
        return False

    # Try available backends: imageio-ffmpeg -> cv2 -> frame sequence
    backend = _get_video_backend()
    if backend == "imageio":
        _write_with_imageio(frames, out_path, fps)
        return True
    elif backend == "cv2":
        _write_with_cv2(frames, out_path, fps)
        return True
    else:
        _write_frame_sequence(frames, out_path)
        return False


def _get_video_backend():
    try:
        import imageio_ffmpeg
        return "imageio"
    except Exception:
        pass
    try:
        import cv2
        return "cv2"
    except Exception:
        pass
    return "frames"


def _write_with_imageio(frames, out_path, fps):
    import imageio
    writer = imageio.get_writer(out_path, mode="I", fps=fps, quality=8)
    for f in frames:
        img = np.array(Image.open(f))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        writer.append_data(img)
    writer.close()


def _write_with_cv2(frames, out_path, fps):
    import cv2
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        writer.write(img)
    writer.release()


def _write_frame_sequence(frames, out_path):
    seq_dir = out_path.replace(".mp4", "_frames")
    os.makedirs(seq_dir, exist_ok=True)
    for i, f in enumerate(frames):
        img = Image.open(f)
        img.save(os.path.join(seq_dir, f"{i:04d}.jpg"))
    logger.warning(
        f"No video backend available (install imageio-ffmpeg or opencv-python). "
        f"Saved frame sequence to {seq_dir}"
    )
    return True


def make_videos_for_scene(scene_dir: str, output_dir: str, cameras: list, fps: int):
    """Create ground-truth videos for one scene."""
    image_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(image_dir):
        logger.warning(f"Images directory not found: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing scene: {scene_dir} -> {output_dir}")

    for cam_id in cameras:
        out_name = f"gt_{CAMERA_NAMES.get(cam_id, f'cam{cam_id}')}.mp4"
        out_path = os.path.join(output_dir, out_name)
        success = make_video_for_camera(image_dir, cam_id, out_path, fps)
        if success:
            logger.info(f"  Saved {out_path}")


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )

    if args.scene_ids is not None:
        scenes = [str(s).zfill(3) for s in args.scene_ids]
    elif args.split_file is not None:
        with open(args.split_file, "r") as fp:
            lines = fp.readlines()[1:]
        scenes = [line.strip().split(",")[0].zfill(3) for line in lines]
    else:
        split_dir = os.path.join(args.processed_dir, args.split)
        scenes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    cameras = args.cameras if args.cameras is not None else list(range(5))

    for scene in scenes:
        scene_dir = os.path.join(args.processed_dir, args.split, scene)
        if not os.path.isdir(scene_dir):
            logger.warning(f"Scene directory not found, skipping: {scene_dir}")
            continue

        if args.output_dir is not None:
            out_dir = os.path.join(args.output_dir, args.split, scene, "videos")
        else:
            out_dir = os.path.join(scene_dir, "videos")

        make_videos_for_scene(scene_dir, out_dir, cameras, args.fps)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch preprocessed Waymo frames into ground-truth MP4 videos.")
    parser.add_argument("--processed_dir", type=str, default="data/waymo/processed", help="Root directory containing processed scene folders.")
    parser.add_argument("--split", type=str, default="training", help="Data split subdirectory (e.g., training, validation).")
    parser.add_argument("--scene_ids", type=int, nargs="+", default=None, help="Specific scene IDs to process (e.g., 23 114).")
    parser.add_argument("--split_file", type=str, default=None, help="Text file listing scene IDs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save videos. If None, saves to <scene_dir>/videos.")
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Camera IDs to include (0-4). Default: all cameras.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    args = parser.parse_args()
    main(args)
