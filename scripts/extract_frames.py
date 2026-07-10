"""Extract frames from video files for annotation and training."""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str = "data/processed/frames",
    fps: float = 2.0,
    resize: tuple = None,
    max_frames: int = 0,
):
    """Extract frames from a video file.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (e.g., 2.0 = 1 frame every 0.5s).
        resize: Optional (width, height) to resize frames.
        max_frames: Max frames to extract (0 = all).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    video_name = video_path.stem
    extracted = 0
    frame_idx = 0

    print(f"Video: {video_path.name}")
    print(f"Video FPS: {video_fps:.1f}, Extract at: {fps} FPS (every {frame_interval} frames)")
    print(f"Total video frames: {total_frames}")

    pbar = tqdm(total=total_frames, desc="Extracting")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            filename = f"{video_name}_{frame_idx:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            extracted += 1

            if max_frames > 0 and extracted >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Extracted {extracted} frames to: {output_dir}")


def extract_from_directory(
    video_dir: str,
    output_dir: str = "data/processed/frames",
    fps: float = 2.0,
    resize: tuple = None,
):
    """Extract frames from all videos in a directory."""
    video_dir = Path(video_dir)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    videos = [f for f in video_dir.iterdir()
              if f.suffix.lower() in video_extensions]

    print(f"Found {len(videos)} videos in {video_dir}")

    for video in videos:
        extract_frames(str(video), output_dir, fps, resize)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--output", default="data/processed/frames")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        help="Resize to width height (e.g., --resize 640 640)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    resize = tuple(args.resize) if args.resize else None

    if input_path.is_dir():
        extract_from_directory(str(input_path), args.output, args.fps, resize)
    else:
        extract_frames(str(input_path), args.output, args.fps, resize, args.max_frames)
