# Custom Scripts

This directory is for ad-hoc utility scripts that are not part of the main training / evaluation pipeline.

## Scripts

- `make_gt_videos.py` — Stitch preprocessed Waymo frames into per-camera ground-truth MP4 videos.

## Usage

Run scripts from the repo root with `PYTHONPATH` set:

```bash
export PYTHONPATH=$(pwd)
python tools/custom/<script_name>.py [args]
```
