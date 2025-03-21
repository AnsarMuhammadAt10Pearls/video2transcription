import os
from pydub import AudioSegment

def convert_video_to_audio(video_path, output_path, output_format="mp3"):
    """
    Convert a video file to an audio file using pydub.
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path for the output audio file
        output_format (str): Format for the output audio
    
    Returns:
        str: Path to the created audio file
    """
    try:
        # Extract audio
        audio = AudioSegment.from_file(video_path)
        
        # Export as audio file
        audio.export(output_path, format=output_format)
        
        print(f"Successfully converted {video_path} to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error converting {video_path} to audio: {str(e)}")
        return None

def main():
    # Hardcoded input filename
    video_path = "sample5.mp4"
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist in the current directory.")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Hardcoded output filename (same name but with mp3 extension)
    output_path = "sample5.mp3"
    
    # Convert the video to audio
    convert_video_to_audio(video_path, output_path)
    
    # Verify the result
    if os.path.exists(output_path):
        print(f"Conversion successful! Audio file saved as {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    else:
        print("Conversion failed. Output file was not created.")

if __name__ == "__main__":
    print("Starting video to audio conversion...")
    main()
    print("Conversion process completed.")