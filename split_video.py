#give me a program that would take a video and split it into 5 minute videos and then copy to a folder named video_splits
# and attach the datetime to the folder name

import os
import datetime
import argparse
import subprocess
from pathlib import Path

def split_video(input_file, segment_length=300):
    """
    Split a video into segments of specified length (default 5 minutes)
    
    Args:
        input_file: Path to the input video file
        segment_length: Length of each segment in seconds (default 300 seconds = 5 minutes)
    """
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"video_splits_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(input_file).stem
    extension = Path(input_file).suffix
    
    # Check for ffmpeg in current directory first
    ffmpeg_cmd = 'ffmpeg'
    if os.path.exists('ffmpeg.exe'):
        ffmpeg_cmd = './ffmpeg.exe'
        print("Using ffmpeg.exe from current directory")
    
    # Use ffmpeg to split the video
    cmd = [
        ffmpeg_cmd,
        '-i', input_file,
        '-c', 'copy',  # Copy without re-encoding
        '-map', '0',
        '-segment_time', str(segment_length),
        '-f', 'segment',
        '-reset_timestamps', '1',
        f'{output_dir}/{base_name}_%03d{extension}'
    ]
    
    print(f"Splitting video into {segment_length} second segments...")
    print(f"Output will be saved to folder: {output_dir}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video successfully split! Files are in the '{output_dir}' folder.")
    except subprocess.CalledProcessError as e:
        print(f"Error during video splitting: {e}")
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your PATH.")
        print("To fix this:")
        print("1. Download ffmpeg from https://ffmpeg.org/download.html")
        print("2. Either add it to your system PATH")
        print("   OR place ffmpeg.exe in the same directory as this script")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into 5-minute segments")
    parser.add_argument("input_file", nargs='?', default="Product_session_17_Version_Repo_by_Eduard_02Dec2022.mp4",
                        help="Path to the input video file (default: Product_session_17_Version_Repo_by_Eduard_02Dec2022.mp4)")
    parser.add_argument("-l", "--length", type=int, default=300, 
                        help="Length of each segment in seconds (default: 300 seconds = 5 minutes)")
    
    args = parser.parse_args()
    split_video(args.input_file, args.length)

