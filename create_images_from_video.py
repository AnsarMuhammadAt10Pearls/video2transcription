import os
import cv2
import base64
import requests
import json
import numpy as np
from dotenv import load_dotenv
import hashlib
import tempfile
import subprocess
import librosa
import matplotlib.pyplot as plt
import math
import argparse
import sys
from PIL import Image
import io
import torch
import uuid
import time
from datetime import datetime, timedelta
import functools
from transformers import (
    AutoModelForObjectDetection, 
    DetrImageProcessor,
    AutoFeatureExtractor, 
    AutoModelForImageClassification,
    AutoTokenizer, 
    pipeline
)

# Create a performance tracker dictionary
performance_tracker = {
    "start_time": None,
    "end_time": None,
    "total_execution_time": 0,  # Total execution time in seconds
    "functions": {}
}

def timing_decorator(func):
    """Decorator to track function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting {func.__name__}... [Time: {datetime.now().strftime('%H:%M:%S')}]")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store the execution time in our tracker
        if func.__name__ in performance_tracker["functions"]:
            performance_tracker["functions"][func.__name__]["count"] += 1
            performance_tracker["functions"][func.__name__]["total_time"] += execution_time
        else:
            performance_tracker["functions"][func.__name__] = {
                "count": 1,
                "total_time": execution_time
            }
        
        # Print execution time
        minutes = execution_time / 60
        print(f"Finished {func.__name__} in {minutes:.2f} minutes ({execution_time:.2f} seconds)")
        
        return result
    return wrapper

# Load environment variables
load_dotenv()

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = input("Please enter your Groq API key: ")

# Global model holders to avoid reloading
models = {
    "object_detection": None,
    "scene_classification": None,
    "image_captioning": None,
    "object_processor": None,
    "scene_processor": None
}

def get_file_info(file_path):
    """Get detailed information about a file"""
    if not os.path.exists(file_path):
        return None
    
    # Get file size
    size = os.path.getsize(file_path)
    
    # Get file hash
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()
    
    # Get video properties if it's a video file
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": file_hash,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": round(duration, 2)
        }
    else:
        return {
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": file_hash
        }

@timing_decorator
def extract_frames(video_path, num_frames=3, output_folder=None):
    """Extract frames from a video file at key timestamps"""
    print(f"Extracting {num_frames} frames from {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = length / fps
    
    frames = []
    frame_positions = []
    
    # Calculate positions to extract frames from (evenly distributed)
    for i in range(num_frames):
        pos = int((i + 1) * length / (num_frames + 1))
        frame_positions.append(pos)
    
    # Extract frames at calculated positions
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # Use output folder if provided
            if output_folder:
                frame_path = os.path.join(output_folder, f"frame_{pos}_{uuid.uuid4().hex[:8]}.jpg")
            else:
                frame_path = f"frame_{pos}_{uuid.uuid4().hex[:8]}.jpg"
                
            cv2.imwrite(frame_path, frame)
            
            # Calculate timestamp in seconds
            timestamp = pos / fps
            
            frames.append({
                "position": pos, 
                "path": frame_path,
                "timestamp": timestamp,
                "timestamp_formatted": format_timestamp(timestamp)
            })
            print(f"  Extracted frame at position {pos} (time: {format_timestamp(timestamp)}) and saved to {frame_path}")
        else:
            print(f"  Failed to extract frame at position {pos}")
    
    cap.release()
    return frames

def format_timestamp(seconds):
    """Format seconds into HH:MM:SS.ms format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def format_time_seconds_to_hms(seconds):
    """Format seconds into HH:MM:SS format"""
    return str(timedelta(seconds=round(seconds)))

def load_object_detection_model():
    """Load the object detection model if not already loaded"""
    if models["object_detection"] is None or models["object_processor"] is None:
        print("Loading object detection model...")
        try:
            # Load DETR model from Hugging Face
            models["object_processor"] = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            models["object_detection"] = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            print("Object detection model loaded successfully")
        except Exception as e:
            print(f"Error loading object detection model: {e}")
            models["object_detection"] = None
            models["object_processor"] = None
    
    return models["object_detection"], models["object_processor"]

def load_scene_classification_model():
    """Load the scene classification model if not already loaded"""
    if models["scene_classification"] is None or models["scene_processor"] is None:
        print("Loading scene classification model...")
        try:
            # Use a general image classification model
            models["scene_processor"] = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
            models["scene_classification"] = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
            print("Scene classification model loaded successfully")
        except Exception as e:
            print(f"Error loading scene classification model: {e}")
            models["scene_classification"] = None
            models["scene_processor"] = None
    
    return models["scene_classification"], models["scene_processor"]

def load_image_captioning_pipeline():
    """Load the image captioning pipeline if not already loaded"""
    if models["image_captioning"] is None:
        print("Loading image captioning model...")
        try:
            # Use BLIP for image captioning
            models["image_captioning"] = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            print("Image captioning model loaded successfully")
        except Exception as e:
            print(f"Error loading image captioning model: {e}")
            models["image_captioning"] = None
    
    return models["image_captioning"]

@timing_decorator
def transcribe_with_groq(audio_path):
    """Transcribe audio using Groq API for analysis"""
    try:
        print("Using Groq API for audio transcription analysis...")
        
        # Get audio duration
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        # If audio is too long, analyze chunks
        max_analysis_duration = 60  # seconds
        
        if duration <= max_analysis_duration:
            # Process whole audio file
            return analyze_audio_content(audio_path)
        else:
            # For longer files, extract multiple segments and analyze each
            segments = []
            full_text = ""
            
            # Extract 3 segments: beginning, middle, end
            segment_positions = [0, duration/2, max(0, duration-max_analysis_duration)]
            
            for i, start_pos in enumerate(segment_positions):
                # Extract segment
                segment_path = f"audio_segment_{i}_{uuid.uuid4().hex[:8]}.wav"
                extract_audio_segment(audio_path, segment_path, start_pos, min(max_analysis_duration, duration-start_pos))
                
                if os.path.exists(segment_path):
                    # Analyze segment
                    segment_result = analyze_audio_content(segment_path, start_pos)
                    if segment_result and "segments" in segment_result:
                        segments.extend(segment_result["segments"])
                        if "full_text" in segment_result:
                            full_text += segment_result["full_text"] + " "
                    
                    # Clean up
                    os.unlink(segment_path)
            
            return {
                "full_text": full_text.strip(),
                "segments": segments,
                "note": "Audio was analyzed in segments due to length"
            }
    
    except Exception as e:
        print(f"Error transcribing with Groq: {str(e)}")
        return None

def extract_audio_segment(audio_path, output_path, start_time, duration):
    """Extract a segment of audio using ffmpeg"""
    try:
        # Find ffmpeg
        ffmpeg_path = 'ffmpeg'
        if os.path.exists('ffmpeg.exe'):
            ffmpeg_path = './ffmpeg.exe'
        
        # Use ffmpeg to extract segment
        command = [
            ffmpeg_path,
            '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        return False
        
    except Exception as e:
        print(f"Error extracting audio segment: {str(e)}")
        return False

@timing_decorator
def analyze_audio_content(audio_path, start_offset=0):
    """Use Groq to analyze audio content and create a pseudo-transcript"""
    try:
        # Convert audio to features for analysis
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        # Extract audio features for description
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
        
        # Create technical description for Groq
        audio_description = f"""
This is a {duration:.2f} second audio clip. Technical analysis shows:
- MFCCs mean: {np.mean(mfccs, axis=1).tolist()}
- Spectral contrast mean: {np.mean(spectral_contrast, axis=1).tolist()}
- Chroma mean: {np.mean(chroma, axis=1).tolist()}
- Tempo: {tempo:.2f} BPM
"""
        
        # Additional energy and rhythm analysis
        rms = librosa.feature.rms(y=audio_data)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonic_ratio = np.sum(np.abs(harmonic)) / np.sum(np.abs(audio_data)) if np.sum(np.abs(audio_data)) > 0 else 0
        
        # Speech probability estimate
        speech_probability = 1 - (harmonic_ratio * (1 - np.mean(zcr)))
        
        # Detect silence
        non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
        silent_duration = duration - sum((end - start) / sample_rate for start, end in non_silent_intervals)
        silence_percentage = (silent_duration / duration) * 100 if duration > 0 else 0
        
        audio_description += f"""
Additional audio features:
- RMS energy (loudness) mean: {np.mean(rms):.4f}, max: {np.max(rms):.4f}
- Zero crossing rate (noisiness): {np.mean(zcr):.4f}
- Spectral centroid (brightness): {np.mean(spectral_centroids):.2f}
- Spectral rolloff: {np.mean(spectral_rolloff):.2f}
- Harmonic ratio: {harmonic_ratio:.4f}
- Speech probability: {speech_probability:.4f}
- Silence percentage: {silence_percentage:.2f}%
- Content type: {'Likely Speech' if speech_probability > 0.6 else 'Likely Music' if speech_probability < 0.4 else 'Mixed Speech and Music'}
"""
        
        # Now use Groq to analyze and create transcript
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert audio analyst who can interpret technical audio features to determine what is happening in audio and produce a transcript-like analysis. You can identify speech content, music, sound effects, and other audio elements from technical features.

When speech is likely present (high speech probability, low harmonic ratio), focus on describing what might be being said based on the audio features.

When music is likely present (low speech probability, high harmonic ratio), describe the type of music, tempo, and mood.

When sound effects are likely present (high zero crossing rate, variable spectral content), describe what might be making these sounds.

Format your response as a well-structured JSON object with "full_text" providing an overall description of what's in the audio, and "segments" containing time-based observations."""
        
        user_prompt = f"""I have an audio clip that I need analyzed. I don't have the actual words, but I have detailed technical features. Please analyze these features and generate a likely description of what's happening in this audio.

{audio_description}

Please create a transcript-like analysis that explains what's likely happening in this audio based solely on the technical features. Format your response as a JSON object with:
1. "full_text": A complete description of what you think is happening in the audio
2. "segments": An array of at least 3-5 objects, each with:
   - "start": Start time in seconds (distribute evenly across the {duration:.2f} seconds)
   - "end": End time in seconds
   - "text": Content description for that segment

For example, you might detect speech patterns, music, silence, sound effects, etc. If speech is likely present, don't make up exact words, but suggest what might be discussed based on the audio patterns.

Respond with valid JSON only."""
        
        payload = {
            "model": "llama3-70b-8192",  # Using Llama 3 for better analysis
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        response_content = result["choices"][0]["message"]["content"]
        transcript_data = json.loads(response_content)
        
        # Apply start_offset to segments if needed
        if start_offset > 0 and "segments" in transcript_data:
            for segment in transcript_data["segments"]:
                if "start" in segment:
                    segment["start"] += start_offset
                if "end" in segment:
                    segment["end"] += start_offset
                # Add formatted timestamps
                segment["start_formatted"] = format_timestamp(segment["start"])
                segment["end_formatted"] = format_timestamp(segment["end"])
        elif "segments" in transcript_data:
            # Add formatted timestamps without offset
            for segment in transcript_data["segments"]:
                if "start" in segment and "end" in segment:
                    segment["start_formatted"] = format_timestamp(segment["start"])
                    segment["end_formatted"] = format_timestamp(segment["end"])
        
        print(f"Audio analysis completed: created transcript-like analysis of {len(transcript_data.get('segments', []))} segments")
        return transcript_data
        
    except Exception as e:
        print(f"Error analyzing audio content: {str(e)}")
        return None

def detect_objects(image_path):
    """Detect objects in an image using DETR model"""
    try:
        model, processor = load_object_detection_model()
        if model is None or processor is None:
            return {"error": "Object detection model not available"}
        
        # Load and process image
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        
        # Format results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "box": box
            })
        
        return {
            "objects": detections,
            "count": len(detections)
        }
    
    except Exception as e:
        print(f"Error in object detection: {e}")
        return {"error": str(e)}

def classify_scene(image_path):
    """Classify the scene in an image"""
    try:
        model, processor = load_scene_classification_model()
        if model is None or processor is None:
            return {"error": "Scene classification model not available"}
        
        # Load and process image
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Get top predictions
        probs = outputs.logits.softmax(dim=1)
        top_probs, top_indices = probs.topk(5)
        
        # Format results
        top_scenes = []
        for score, idx in zip(top_probs[0], top_indices[0]):
            label = model.config.id2label[idx.item()]
            top_scenes.append({
                "scene": label,
                "confidence": round(score.item(), 3)
            })
        
        return {
            "scenes": top_scenes
        }
    
    except Exception as e:
        print(f"Error in scene classification: {e}")
        return {"error": str(e)}

def generate_image_caption(image_path):
    """Generate a descriptive caption for an image"""
    try:
        captioner = load_image_captioning_pipeline()
        if captioner is None:
            return {"error": "Image captioning model not available"}
        
        # Generate caption
        caption = captioner(image_path)
        
        return {
            "caption": caption[0]["generated_text"] if caption else "No caption generated"
        }
    
    except Exception as e:
        print(f"Error generating image caption: {e}")
        return {"error": str(e)}

@timing_decorator
def analyze_frame_semantic(frame_path):
    """Perform semantic analysis on a frame including objects, scene, and caption"""
    print(f"Performing semantic analysis on frame: {frame_path}")
    
    results = {
        "objects": detect_objects(frame_path),
        "scene": classify_scene(frame_path),
        "caption": generate_image_caption(frame_path),
    }
    
    # Also include the technical analysis
    technical = analyze_frame_technical(frame_path)
    results["technical"] = technical
    
    return results

def analyze_frame_technical(frame_path):
    """Analyze a single frame using OpenCV for technical features"""
    try:
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return {"error": "Could not read frame"}
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert to grayscale for some analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic image analysis
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # Detect faces (if any)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Build description
        technical_data = {
            "resolution": {
                "width": width,
                "height": height
            },
            "brightness": {
                "value": round(float(brightness), 2),
                "level": 'High' if brightness > 170 else 'Medium' if brightness > 85 else 'Low'
            },
            "contrast": {
                "value": round(float(contrast), 2),
                "level": 'High' if contrast > 60 else 'Medium' if contrast > 30 else 'Low'
            },
            "edge_density": {
                "value": round(float(edge_density), 4),
                "level": 'High' if edge_density > 0.1 else 'Medium' if edge_density > 0.05 else 'Low'
            },
            "faces_detected": len(faces)
        }
        
        # Add color analysis
        b, g, r = cv2.split(frame)
        rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
        
        technical_data["color"] = {
            "rgb_mean": [round(float(c), 2) for c in rgb_mean],
            "dominant_channel": ["Red", "Green", "Blue"][np.argmax(rgb_mean)]
        }
        
        # Add face locations if detected
        if len(faces) > 0:
            technical_data["faces"] = []
            for (x, y, w, h) in faces:
                technical_data["faces"].append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
        
        return technical_data
        
    except Exception as e:
        return {"error": f"Error analyzing frame: {str(e)}"}

@timing_decorator
def extract_audio_for_transcription(video_path):
    """Extract audio from video file for transcription"""
    print("Extracting audio for transcription...")
    
    try:
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        print(f"Temporary audio file created at: {temp_audio_path}")
        
        # Find ffmpeg
        ffmpeg_path = 'ffmpeg'
        if os.path.exists('ffmpeg.exe'):
            ffmpeg_path = './ffmpeg.exe'
        
        # Use ffmpeg to extract audio
        command = [
            ffmpeg_path, 
            '-i', video_path, 
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            temp_audio_path
        ]
        
        print(f"Executing ffmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verify file exists
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            print(f"Audio file created successfully ({os.path.getsize(temp_audio_path)} bytes)")
            return temp_audio_path
        else:
            print("Failed to extract audio: output file is empty or missing")
            return None
            
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Groq-based analysis"""
    return transcribe_with_groq(audio_path)

@timing_decorator
def extract_audio_features(video_path):
    """Extract technical audio features for analysis"""
    print("Extracting audio features...")
    
    try:
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Find ffmpeg
        ffmpeg_path = 'ffmpeg'
        if os.path.exists('ffmpeg.exe'):
            ffmpeg_path = './ffmpeg.exe'
            
        # Use ffmpeg to extract audio
        command = [
            ffmpeg_path, 
            '-i', video_path, 
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            '-y',
            temp_audio_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Load audio with librosa
        audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
        
        # Calculate audio features
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        # RMS energy (loudness)
        rms = librosa.feature.rms(y=audio_data)[0]
        avg_loudness = np.mean(rms)
        max_loudness = np.max(rms)
        
        # Extract tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        
        # Zero crossing rate (indicator of noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        avg_zcr = np.mean(zcr)
        
        # Spectral centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)
        
        # Get harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonic_ratio = np.sum(np.abs(harmonic)) / np.sum(np.abs(audio_data)) if np.sum(np.abs(audio_data)) > 0 else 0
        
        # Detect silence
        non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
        silent_duration = duration - sum((end - start) / sample_rate for start, end in non_silent_intervals)
        silence_percentage = (silent_duration / duration) * 100 if duration > 0 else 0
        
        # MFCC features (often used for speech recognition)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        avg_rolloff = np.mean(spectral_rolloff)
        
        # Speech probability estimate
        speech_probability = 1 - (harmonic_ratio * (1 - avg_zcr))
        
        # Clean up
        os.unlink(temp_audio_path)
        
        # Return features
        audio_features = {
            "duration_seconds": round(duration, 2),
            "tempo_bpm": round(tempo, 2),
            "average_loudness": round(float(avg_loudness), 4),
            "max_loudness": round(float(max_loudness), 4),
            "zero_crossing_rate": round(float(avg_zcr), 4),
            "spectral_centroid": round(float(avg_spectral_centroid), 2),
            "harmonic_ratio": round(float(harmonic_ratio), 4),
            "silence_percentage": round(silence_percentage, 2),
            "speech_probability": round(float(speech_probability), 4),
            "spectral_rolloff": round(float(avg_rolloff), 2),
            "content_type": 'Likely Speech' if speech_probability > 0.6 else 'Likely Music' if speech_probability < 0.4 else 'Mixed Speech and Music'
        }
        
        return audio_features
        
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return {
            "error": str(e),
            "duration_seconds": 0
        }

@timing_decorator
def analyze_video_with_groq(video_path, frames, transcript, audio_features):
    """Analyze video using Groq for comprehensive understanding"""
    print("Analyzing video with Groq (text-only approach)...")
    
    # API endpoint
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create system prompt
    system_prompt = """You are an expert video analyst tasked with understanding video content based on extracted data. 
Your analysis should focus on:
1. The video's purpose and main content
2. Target audience and engagement strategy
3. Narrative structure and messaging
4. Overall effectiveness and impact

Produce a detailed, insightful analysis based on the frame descriptions, transcript, and audio features provided.
Focus primarily on answering: What is this video about and what is its purpose?"""
    
    # Create user message with detailed information
    user_message = "I need to understand what this video is about. Here's the extracted data:\n\n"
    
    # Add video metadata
    file_info = get_file_info(video_path)
    user_message += f"## Video Metadata\n"
    user_message += f"- Resolution: {file_info.get('width', 'unknown')}x{file_info.get('height', 'unknown')}\n"
    user_message += f"- Duration: {file_info.get('duration_seconds', 'unknown')} seconds\n"
    user_message += f"- FPS: {file_info.get('fps', 'unknown')}\n\n"
    
    # Add frame analysis
    user_message += "## Frame Analysis\n\n"
    for i, frame in enumerate(frames):
        user_message += f"### Frame {i+1} - Timestamp: {frame['timestamp_formatted']}\n"
        
        # Add caption
        if 'caption' in frame['analysis'] and 'caption' in frame['analysis']['caption']:
            user_message += f"**Caption:** {frame['analysis']['caption']['caption']}\n\n"
        
        # Add object detection results
        if 'objects' in frame['analysis'] and 'objects' in frame['analysis']['objects']:
            objects = frame['analysis']['objects']['objects']
            if objects:
                user_message += "**Objects detected:**\n"
                for obj in objects:
                    user_message += f"- {obj['label']} (confidence: {obj['confidence']})\n"
                user_message += "\n"
        
        # Add scene classification
        if 'scene' in frame['analysis'] and 'scenes' in frame['analysis']['scene']:
            scenes = frame['analysis']['scene']['scenes']
            if scenes:
                user_message += "**Scene classification:**\n"
                for scene in scenes[:3]:  # Top 3 scenes
                    user_message += f"- {scene['scene']} (confidence: {scene['confidence']})\n"
                user_message += "\n"
        
        # Add technical details
        if 'technical' in frame['analysis']:
            tech = frame['analysis']['technical']
            user_message += "**Technical details:**\n"
            user_message += f"- Resolution: {tech.get('resolution', {}).get('width', 'unknown')}x{tech.get('resolution', {}).get('height', 'unknown')}\n"
            user_message += f"- Brightness: {tech.get('brightness', {}).get('level', 'unknown')}\n"
            user_message += f"- Contrast: {tech.get('contrast', {}).get('level', 'unknown')}\n"
            user_message += f"- Dominant color: {tech.get('color', {}).get('dominant_channel', 'unknown')}\n"
            user_message += f"- Faces detected: {tech.get('faces_detected', 'unknown')}\n\n"
    
    # Add transcript
    if transcript and 'full_text' in transcript:
        user_message += "## Audio Content Analysis\n\n"
        user_message += transcript['full_text'] + "\n\n"
        
        # Add some key segments with timestamps
        if 'segments' in transcript and len(transcript['segments']) > 0:
            user_message += "### Key Segments with Timestamps\n"
            # Take a few evenly distributed segments
            segments = transcript['segments']
            num_samples = min(5, len(segments))
            step = max(1, len(segments) // num_samples)
            
            for i in range(0, len(segments), step):
                if i < len(segments):
                    segment = segments[i]
                    user_message += f"- {segment.get('start_formatted', 'unknown')} to {segment.get('end_formatted', 'unknown')}: {segment.get('text', 'unknown')}\n"
            user_message += "\n"
    
    # Add audio features
    if audio_features:
        user_message += "## Audio Technical Analysis\n"
        user_message += f"- Duration: {audio_features.get('duration_seconds', 'unknown')} seconds\n"
        user_message += f"- Content Type: {audio_features.get('content_type', 'unknown')}\n"
        user_message += f"- Speech Probability: {audio_features.get('speech_probability', 'unknown')}\n"
        user_message += f"- Tempo: {audio_features.get('tempo_bpm', 'unknown')} BPM\n"
        user_message += f"- Average Loudness: {audio_features.get('average_loudness', 'unknown')}\n"
        user_message += f"- Harmonic Ratio: {audio_features.get('harmonic_ratio', 'unknown')}\n"
        user_message += f"- Silence Percentage: {audio_features.get('silence_percentage', 'unknown')}%\n\n"
    
    # Final instructions for format and focus
    user_message += """
Based on the above information, please provide a comprehensive analysis of the video that includes:

1. **Main Purpose and Content:** What is this video about and what is its primary purpose?
2. **Target Audience:** Who is the likely audience for this content?
3. **Narrative Structure:** How is the content organized and presented?
4. **Key Elements:** What are the most important visual and audio elements?
5. **Engagement Strategy:** How does the video attempt to engage viewers?
6. **Overall Effectiveness:** How successful is the video likely to be at achieving its purpose?

Your response should be structured as a cohesive analysis rather than just answering these questions separately. Focus on insights about the video's purpose and effectiveness."""
    
    # Try to use Llama 3 if available (better quality), fallback to Mixtral
    models_to_try = ["llama3-70b-8192", "mixtral-8x7b-32768"]
    
    for model_name in models_to_try:
        try:
            # Construct messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            # Make the API request
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]
            
            # Get token usage information
            usage = result.get("usage", {})
            token_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
            print(f"\n--- Groq Analysis with {model_name} ---")
            print(analysis[:200] + "..." if len(analysis) > 200 else analysis)
            print(f"Token usage: {token_info}")
            
            return {
                "analysis": analysis,
                "token_usage": token_info,
                "model_used": model_name
            }
            
        except Exception as e:
            print(f"Error using {model_name}: {e}")
            if 'response' in locals():
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text[:500]}")
    
    # If all models failed, return error
    return {
        "error": "All model attempts failed"
    }

@timing_decorator
def analyze_frames_batch(frames):
    """Batch process all frame analyses"""
    print(f"\n=== Analyzing {len(frames)} Frames ===")
    for i, frame in enumerate(frames):
        print(f"Analyzing frame {i+1}/{len(frames)}...")
        frame['analysis'] = analyze_frame_semantic(frame['path'])
    return frames

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Video Analysis with Semantic Understanding")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to extract (default: 5)")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio processing")
    parser.add_argument("--skip-transcription", action="store_true", help="Skip audio transcription")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--keep-images", action="store_true", help="Keep extracted frames in a timestamped folder")
    args = parser.parse_args()
    
    # Start timing for entire process
    start_time = time.time()
    performance_tracker["start_time"] = start_time
    
    print(f"Analysis started at: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    video_path = args.video
    output_path = args.output or f"{os.path.splitext(video_path)[0]}_analysis.json"
    
    # Create a timestamped folder for frames
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    images_folder = os.path.join("images", f"{video_name}_{current_timestamp}")
    
    # Always create the images folder
    os.makedirs(images_folder, exist_ok=True)
    print(f"Created images folder: {images_folder}")
    
    # Print video file information
    print("\n=== Video File Information ===")
    file_info = get_file_info(video_path)
    if file_info:
        print(f"File: {video_path}")
        print(f"Size: {file_info['size_mb']} MB")
        print(f"Resolution: {file_info.get('width', 'unknown')}x{file_info.get('height', 'unknown')}")
        print(f"Duration: {file_info.get('duration_seconds', 'unknown')} seconds")
    else:
        print(f"Error: File {video_path} does not exist.")
        exit()
    print("===========================\n")
    
    # Extract frames
    frames = extract_frames(video_path, num_frames=args.frames, output_folder=images_folder)
    if not frames:
        print("No frames extracted. Exiting.")
        exit()
    
    # Save frame information to a text file in the images folder
    frames_info_path = os.path.join(images_folder, "frames_info.txt")
    with open(frames_info_path, 'w', encoding='utf-8') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis timestamp: {current_timestamp}\n\n")
        for i, frame in enumerate(frames):
            f.write(f"Frame {i+1}:\n")
            f.write(f"  Position: {frame['position']}\n")
            f.write(f"  Timestamp: {frame['timestamp_formatted']}\n")
            f.write(f"  Path: {frame['path']}\n\n")
    
    # Analyze frames
    frames = analyze_frames_batch(frames)
    
    # Process audio
    audio_features = None
    transcript = None
    
    if not args.skip_audio:
        # Extract audio features
        print("\n=== Analyzing Audio ===")
        audio_features = extract_audio_features(video_path)
        
        # Transcribe audio if not skipped
        if not args.skip_transcription:
            print("\n=== Analyzing Audio Content ===")
            audio_path = extract_audio_for_transcription(video_path)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                # Make sure to clean up
                try:
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                except:
                    pass
    
    # Analyze video with Groq
    print("\n=== Generating Video Analysis ===")
    analysis_results = analyze_video_with_groq(video_path, frames, transcript, audio_features)
    
    # Record end time for entire process
    end_time = time.time()
    performance_tracker["end_time"] = end_time
    
    # Calculate total execution time
    total_execution_time = end_time - start_time
    performance_tracker["total_execution_time"] = total_execution_time
    
    # Format ending timestamp
    end_timestamp = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Analysis completed at: {end_timestamp}")
    
    # Prepare performance data for output
    performance_data = {
        "start_time": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_timestamp,
        "total_execution_time_seconds": total_execution_time,
        "total_execution_time_minutes": total_execution_time / 60,
        "total_execution_time_formatted": format_time_seconds_to_hms(total_execution_time),
        "function_timings": []
    }
    
    # Sort functions by execution time (descending)
    sorted_functions = sorted(
        performance_tracker["functions"].items(), 
        key=lambda x: x[1]["total_time"], 
        reverse=True
    )
    
    # Add top functions to performance data
    for func_name, timing in sorted_functions:
        performance_data["function_timings"].append({
            "function": func_name,
            "execution_count": timing["count"],
            "total_time_seconds": timing["total_time"],
            "total_time_minutes": timing["total_time"] / 60,
            "total_time_formatted": format_time_seconds_to_hms(timing["total_time"]),
            "average_time_seconds": timing["total_time"] / timing["count"],
            "average_time_minutes": (timing["total_time"] / timing["count"]) / 60,
            "percentage_of_total": (timing["total_time"] / total_execution_time) * 100 if total_execution_time > 0 else 0
        })
    
    # Prepare final output
    output_data = {
        "video_info": file_info,
        "frames": [
            {
                "position": frame["position"],
                "timestamp": frame["timestamp"],
                "timestamp_formatted": frame["timestamp_formatted"],
                "analysis": frame["analysis"],
                "path": frame["path"]
            } for frame in frames
        ],
        "audio_features": audio_features,
        "transcript": transcript,
        "analysis": analysis_results,
        "performance": performance_data,
        "images_folder": images_folder
    }
    
    # Save results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAnalysis saved to {output_path}")
    
    # Print final summary
    print("\n=== Analysis Summary ===")
    print("\nGroq Analysis:")
    groq_analysis = analysis_results.get("analysis", "")
    print(groq_analysis[:500] + "..." if len(groq_analysis) > 500 else groq_analysis)
    
    # Print performance statistics
    print("\n=== Performance Statistics ===")
    print(f"Started:  {performance_data['start_time']}")
    print(f"Finished: {performance_data['end_time']}")
    print(f"Total execution time: {performance_data['total_execution_time_minutes']:.2f} minutes ({performance_data['total_execution_time_formatted']})")
    print("\nTop 5 most time-consuming functions:")
    
    for i, func_data in enumerate(performance_data["function_timings"][:5]):
        func_name = func_data["function"]
        time_mins = func_data["total_time_minutes"]
        time_formatted = func_data["total_time_formatted"]
        count = func_data["execution_count"]
        percentage = func_data["percentage_of_total"]
        
        print(f"{i+1}. {func_name}: {time_mins:.2f} minutes ({time_formatted}), {percentage:.1f}% of total time, called {count} times")
    
    # Save performance data to the images folder
    performance_path = os.path.join(images_folder, "performance_stats.txt")
    with open(performance_path, 'w', encoding='utf-8') as f:
        f.write("=== Performance Statistics ===\n")
        f.write(f"Started:  {performance_data['start_time']}\n")
        f.write(f"Finished: {performance_data['end_time']}\n")
        f.write(f"Total execution time: {performance_data['total_execution_time_minutes']:.2f} minutes ({performance_data['total_execution_time_formatted']})\n\n")
        f.write("Top functions by execution time:\n")
        
        for i, func_data in enumerate(performance_data["function_timings"]):
            func_name = func_data["function"]
            time_mins = func_data["total_time_minutes"]
            time_formatted = func_data["total_time_formatted"]
            count = func_data["execution_count"]
            percentage = func_data["percentage_of_total"]
            
            f.write(f"{i+1}. {func_name}: {time_mins:.2f} minutes ({time_formatted}), {percentage:.1f}% of total time, called {count} times\n")
    
    print(f"\nImages saved to: {images_folder}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()