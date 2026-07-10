"""Download free test traffic videos for the project.

Usage:
    python scripts/download_test_videos.py

Downloads sample Indian traffic videos from YouTube (royalty-free)
using yt-dlp which is already in requirements.txt.
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("data/raw/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Free Indian traffic footage (short clips, public domain / Creative Commons)
VIDEOS = [
    {
        "name": "indian_traffic_01",
        "url": "https://www.youtube.com/watch?v=MNn9qKG2UFI",
        "desc": "Indian road traffic - Bangalore",
    },
    {
        "name": "indian_traffic_02",
        "url": "https://www.youtube.com/watch?v=7HaJArMDKgI",
        "desc": "Heavy traffic in India",
    },
    {
        "name": "indian_traffic_03",
        "url": "https://www.youtube.com/watch?v=jjlBnrzSGjc",
        "desc": "Indian highway traffic",
    },
]


def download_video(name: str, url: str, desc: str):
    output_path = OUTPUT_DIR / f"{name}.mp4"

    if output_path.exists():
        print(f"[SKIP] {name} already exists.")
        return

    print(f"\n[DOWNLOADING] {desc}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "-o", str(output_path),
        "-f", "best[height<=720][ext=mp4]",
        "--no-playlist",
        "--max-filesize", "50M",
        "--socket-timeout", "30",
    ]

    try:
        subprocess.run(cmd, check=True, timeout=120)
        print(f"[DONE] {name} downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {name}: {e}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} download took too long.")


def create_synthetic_test_video():
    """Create a simple synthetic test video with moving rectangles (no download needed)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[SKIP] OpenCV not installed. Cannot create synthetic video.")
        return

    output_path = OUTPUT_DIR / "synthetic_traffic.mp4"
    if output_path.exists():
        print(f"[SKIP] synthetic_traffic.mp4 already exists.")
        return

    print("\n[CREATING] Synthetic test video with moving vehicles...")

    width, height = 1280, 720
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Simulated vehicles (x, y, w, h, speed_x, speed_y, color)
    vehicles = [
        [100, 300, 80, 50, 4, 0, (0, 255, 0)],     # Car going right
        [1100, 400, 80, 50, -3, 0, (255, 0, 0)],    # Car going left
        [600, 100, 60, 40, 0, 3, (0, 0, 255)],      # Car going down
        [400, 600, 60, 40, 0, -2, (255, 255, 0)],   # Car going up
        [200, 500, 120, 60, 5, 0, (0, 165, 255)],   # Bus going right
        [900, 250, 50, 30, -6, 0, (128, 0, 128)],   # Bike going left
    ]

    for frame_idx in range(total_frames):
        # Background - road
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray road

        # Draw road lines
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        for x in range(0, width, 40):
            cv2.line(frame, (x, height // 3), (x + 20, height // 3), (200, 200, 200), 1)
            cv2.line(frame, (x, 2 * height // 3), (x + 20, 2 * height // 3), (200, 200, 200), 1)

        # Draw and move vehicles
        for v in vehicles:
            x, y, w, h, sx, sy, color = v
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)

            # Move vehicle
            v[0] += sx
            v[1] += sy

            # Wrap around
            if v[0] > width + 50:
                v[0] = -w
            elif v[0] < -w - 50:
                v[0] = width
            if v[1] > height + 50:
                v[1] = -h
            elif v[1] < -h - 50:
                v[1] = height

        # Add timestamp text
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "SYNTHETIC TEST VIDEO", (width // 2 - 150, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        writer.write(frame)

    writer.release()
    print(f"[DONE] Synthetic video saved: {output_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Duration: {duration}s")


if __name__ == "__main__":
    print("=" * 60)
    print("  Test Video Downloader / Generator")
    print("=" * 60)

    # Always create synthetic video (works without internet)
    create_synthetic_test_video()

    # Try downloading real videos
    print("\n" + "=" * 60)
    print("  Downloading real traffic videos from YouTube...")
    print("=" * 60)

    for video in VIDEOS:
        download_video(**video)

    print("\n" + "=" * 60)
    print("  All done! Videos saved in: data/raw/videos/")
    print("=" * 60)
