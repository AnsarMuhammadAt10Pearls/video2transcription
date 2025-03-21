import speech_recognition as sr
from pydub import AudioSegment
import os
import math
import time
from pydub.silence import split_on_silence

def transcribe_audio():
    # File path
    mp3_file = "sample5.mp3"
    
    # Check if file exists
    if not os.path.exists(mp3_file):
        print(f"Error: The file {mp3_file} does not exist.")
        return
    
    print(f"Converting {mp3_file} to WAV format...")
    
    # Convert mp3 to wav (speech_recognition works with wav files)
    sound = AudioSegment.from_mp3(mp3_file)
    wav_file = "temp_audio.wav"
    sound.export(wav_file, format="wav")
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Adjust recognition parameters for better accuracy
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Longer pause threshold for better sentence breaks
    
    # Get audio duration in seconds
    audio_duration = len(sound) / 1000
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Process long audio in smaller chunks for better accuracy
    chunk_size = 15  # seconds (reduced from 30 for better performance)
    full_transcript = ""
    
    # Create temp directory for audio chunks
    temp_dir = "temp_audio_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Option 1: Process audio based on silence detection
        print("Processing audio using silence detection...")
        chunks = split_on_silence(
            sound,
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=sound.dBFS-16,  # silence threshold
            keep_silence=500  # keep some silence for natural breaks
        )
        
        print(f"Audio split into {len(chunks)} chunks based on silence detection")
        
        for i, chunk in enumerate(chunks):
            # Export chunk and save it
            chunk_filename = os.path.join(temp_dir, f"chunk{i}.wav")
            chunk.export(chunk_filename, format="wav")
            
            # Process this chunk
            with sr.AudioFile(chunk_filename) as source:
                audio = recognizer.record(source)
                
                try:
                    # Try with Google's API
                    chunk_text = recognizer.recognize_google(audio, language="en-US", show_all=False)
                    full_transcript += chunk_text + " "
                    print(f"Chunk {i+1}/{len(chunks)}: Transcribed successfully")
                except sr.UnknownValueError:
                    print(f"Chunk {i+1}/{len(chunks)}: Failed to recognize speech")
                except sr.RequestError as e:
                    print(f"Chunk {i+1}/{len(chunks)}: API request error: {e}")
                    # Add a delay to avoid overwhelming the API
                    time.sleep(0.5)
        
        # Option 2 (backup): If silence detection yields poor results, process in fixed time chunks
        if not full_transcript.strip():
            print("Silence detection method gave no results. Trying fixed time chunks...")
            full_transcript = ""  # Reset transcript
            
            with sr.AudioFile(wav_file) as source:
                # Calculate number of chunks
                num_chunks = math.ceil(audio_duration / chunk_size)
                print(f"Processing audio in {num_chunks} fixed chunks...")
                
                for i in range(num_chunks):
                    print(f"Processing chunk {i+1}/{num_chunks}...")
                    # Calculate start and end positions for this chunk
                    start_pos = i * chunk_size
                    end_pos = min((i + 1) * chunk_size, audio_duration)
                    
                    # Set audio position and read chunk
                    source.stream.seek(int(start_pos * source.SAMPLE_RATE * source.SAMPLE_WIDTH))
                    chunk_audio = recognizer.record(source, duration=min(chunk_size, end_pos - start_pos))
                    
                    # Try with different recognition services for better results
                    try:
                        chunk_text = recognizer.recognize_google(chunk_audio, language="en-US", show_all=False)
                        full_transcript += chunk_text + " "
                        print(f"Chunk {i+1}/{num_chunks}: Transcribed successfully")
                    except sr.UnknownValueError:
                        print(f"Chunk {i+1}/{num_chunks}: Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Chunk {i+1}/{num_chunks}: API request error: {e}")
                    
                    # Add a short delay to avoid API rate limits
                    time.sleep(0.5)
            
        print("\nFull Transcription:")
        print(full_transcript.strip())
        
        # Save transcript to file
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(full_transcript.strip())
            print("Transcript saved to transcript.txt")
                
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        # Clean up temp files
        if os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"Removed temporary file: {wav_file}")
        
        # Clean up chunk files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print("Removed temporary chunk files")

if __name__ == "__main__":
    transcribe_audio()