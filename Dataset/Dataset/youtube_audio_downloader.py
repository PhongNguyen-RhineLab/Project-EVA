#!/usr/bin/env python3
"""
Simple YouTube Audio Downloader
Downloads audio from YouTube videos in MP3 format
"""

import sys

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not found. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "--break-system-packages"])
    import yt_dlp


def download_audio(url, output_path="downloads"):
    """
    Download audio from a YouTube URL

    Args:
        url: YouTube video URL
        output_path: Directory to save the audio file
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {url}")

            # First, get video info to check available formats
            info = ydl.extract_info(url, download=False)

            # Check if there are any audio formats available
            if not info.get('formats'):
                print("No downloadable formats found. The video might be restricted.")
                return

            # Try to download
            ydl.download([url])
            print("Download complete!")

    except yt_dlp.utils.DownloadError as e:
        print(f"\nDownload failed: {e}")
        print("\nTrying alternative method...")

        # Try with different options
        ydl_opts_alt = {
            'format': 'worstaudio/worst',  # Try lower quality
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'quiet': False,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_alt) as ydl:
                ydl.download([url])
                print("Download complete with alternative method!")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("\nPossible solutions:")
            print("1. Update yt-dlp: pip install -U yt-dlp")
            print("2. The video might be age-restricted or region-locked")
            print("3. Try a different video")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_audio_downloader.py <youtube_url>")
        print("Example: python youtube_audio_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        sys.exit(1)

    youtube_url = sys.argv[1]
    download_audio(youtube_url)