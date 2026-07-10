"""Download traffic videos from YouTube for dataset creation."""

import argparse
from pathlib import Path

# Video URLs should be provided via a text file or command line
# These are placeholder search terms - user must provide actual URLs
SAMPLE_SEARCH_TERMS = [
    "Indian traffic CCTV footage",
    "Bangalore traffic camera live",
    "highway traffic India surveillance",
    "intersection traffic monitoring camera",
]


def download_videos(
    urls_file: str = None,
    urls: list = None,
    output_dir: str = "data/raw/youtube_downloads",
    resolution: str = "720",
):
    """Download videos from YouTube using yt-dlp.

    Args:
        urls_file: Text file with one URL per line.
        urls: List of YouTube URLs.
        output_dir: Output directory.
        resolution: Max resolution (e.g., "720", "1080").
    """
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not installed. Run: pip install yt-dlp")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect URLs
    all_urls = []
    if urls:
        all_urls.extend(urls)
    if urls_file:
        with open(urls_file) as f:
            all_urls.extend(line.strip() for line in f if line.strip())

    if not all_urls:
        print("No URLs provided.")
        print("Usage:")
        print("  python download_youtube_videos.py --urls-file urls.txt")
        print("  python download_youtube_videos.py --url 'https://youtube.com/...'")
        print(f"\nSearch terms for finding traffic videos: {SAMPLE_SEARCH_TERMS}")
        return

    ydl_opts = {
        "format": f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
    }

    print(f"Downloading {len(all_urls)} videos to {output_dir}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, url in enumerate(all_urls, 1):
            print(f"\n[{i}/{len(all_urls)}] Downloading: {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nDownload complete. Videos saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube traffic videos")
    parser.add_argument("--urls-file", help="Text file with URLs (one per line)")
    parser.add_argument("--url", action="append", dest="urls", help="YouTube URL")
    parser.add_argument("--output", default="data/raw/youtube_downloads")
    parser.add_argument("--resolution", default="720")
    args = parser.parse_args()

    download_videos(
        urls_file=args.urls_file,
        urls=args.urls,
        output_dir=args.output,
        resolution=args.resolution,
    )
