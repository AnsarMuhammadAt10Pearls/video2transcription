import os
import glob
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import time
from tabulate import tabulate
import concurrent.futures
import pytesseract
from io import BytesIO
import cv2
import hashlib

# Load environment variables
load_dotenv()

# Query string
QUERY = "Complete Journey of Committing and Deploying of a Symbox Component"

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Groq models to use (updated to only include available models)
GROQ_MODELS = [
    {"name": "llama3-8b-8192", "description": "Llama-3 8B", "temp": 0.1},
    {"name": "llama3-70b-8192", "description": "Llama-3 70B", "temp": 0.1},
]

# Configure path to tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'd:\tesseract\tesseract.exe'

# Create a cache directory for OCR results
OCR_CACHE_DIR = "ocr_cache"
os.makedirs(OCR_CACHE_DIR, exist_ok=True)

def extract_text_from_image(image_path):
    """Extract text from image using OCR with caching to avoid repeated processing"""
    # Generate a hash of the image file
    with open(image_path, 'rb') as img_file:
        img_hash = hashlib.md5(img_file.read()).hexdigest()
    
    cache_file = os.path.join(OCR_CACHE_DIR, f"{img_hash}.txt")
    
    # Check if we have this image's OCR results in cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Process the image with OCR
    try:
        # Use CV2 to preprocess the image for better OCR results
        img = cv2.imread(image_path)
        if img is None:
            # If cv2 fails, try with PIL
            with Image.open(image_path) as pil_img:
                # Convert PIL image to cv2 format
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Apply preprocessing to improve OCR performance
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black and white image
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Perform OCR on both the original grayscale and the threshold version
        text_gray = pytesseract.image_to_string(gray)
        text_threshold = pytesseract.image_to_string(threshold)
        
        # Combine results for higher accuracy (usually one or the other works better)
        combined_text = text_gray + "\n" + text_threshold
        
        # Cache the results
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        return combined_text
    except Exception as e:
        print(f"OCR error for {image_path}: {str(e)}")
        return ""

def extract_image_metadata(image_path):
    """Extract meaningful metadata from an image when possible"""
    try:
        with Image.open(image_path) as img:
            # Get EXIF data
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                for tag, value in img._getexif().items():
                    if tag in TAGS:
                        exif_data[TAGS[tag]] = value
                        
            # Extract description or title if available
            description = exif_data.get('ImageDescription', '')
            
            # Basic image properties
            properties = {
                'format': img.format,
                'size': img.size,
                'mode': img.mode,
                'description': description,
                'filename': os.path.basename(image_path)
            }
            
            return properties
    except Exception as e:
        # If any error occurs, return minimal information
        return {'format': 'unknown', 'error': str(e), 'filename': os.path.basename(image_path)}

def create_image_embedding(image_path, query):
    """Create a semantic embedding for an image based on its filename, metadata, and OCR text"""
    # Extract image filename and path components
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Clean up the name by replacing separators with spaces
    cleaned_name = name_without_ext.replace('_', ' ').replace('-', ' ')
    
    # Try to get image metadata
    try:
        metadata = extract_image_metadata(image_path)
        img_desc = metadata.get('description', '')
    except:
        img_desc = ""
    
    # Extract text using OCR
    ocr_text = extract_text_from_image(image_path)
    
    # Look for exact phrase matches in OCR text
    exact_match_boost = ""
    if query.lower() in ocr_text.lower():
        exact_match_boost = f"EXACT MATCH: {query}. "
    
    # Combine all relevant text for embedding
    combined_text = f"{exact_match_boost}Image: {cleaned_name}. {img_desc}. Text content: {ocr_text}"
    
    # Generate embedding
    return embeddings_model.embed_query(combined_text), ocr_text, exact_match_boost != ""

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def find_matching_images(query, image_folder, min_similarity_threshold=0.2):
    """Find images matching the query using embeddings and OCR"""
    # Make sure the images directory exists
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' not found.")
        return []
    
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query)
    
    # Get all images in the folder
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"No images found in folder '{image_folder}'.")
        return []
    
    print(f"Found {len(image_files)} images. Extracting text (OCR) and calculating similarities...")
    
    # Process images with OCR and calculate similarity
    results = []
    
    # Process in batches for better progress reporting
    batch_size = 5
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({i+1}-{min(i+batch_size, len(image_files))} of {len(image_files)} images)")
        
        for img_path in batch:
            # Generate embedding for the image that includes OCR text
            img_embedding, ocr_text, has_exact_match = create_image_embedding(img_path, query)
            
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, img_embedding)
            
            # Apply boost for exact text matches
            boosted_similarity = similarity
            if has_exact_match:
                # Boost similarity for images with exact phrase matches
                boosted_similarity = min(1.0, similarity * 1.5)  # Boost by 50%, but cap at 1.0
                print(f"  Found exact match in {os.path.basename(img_path)}! Boosting similarity from {similarity:.4f} to {boosted_similarity:.4f}")
            
            # Include all images with similarity score for analysis
            results.append({
                'path': img_path,
                'similarity': boosted_similarity,
                'original_similarity': similarity,
                'filename': os.path.basename(img_path),
                'ocr_text': ocr_text[:500] + ("..." if len(ocr_text) > 500 else ""),  # Truncate very long OCR text
                'has_exact_match': has_exact_match,
                'models_evaluation': []  # Will store all model evaluations
            })
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Only return the top N images for model evaluation
    top_n = 10  # Limit to only process top 10 images
    filtered_results = [img for img in results[:top_n] if img['similarity'] >= min_similarity_threshold]
    
    # Always include exact matches, even if they somehow didn't make the similarity threshold
    exact_matches = [img for img in results if img['has_exact_match'] and img not in filtered_results]
    filtered_results.extend(exact_matches[:5])  # Add up to 5 additional exact matches
    
    # Re-sort after adding exact matches
    filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    if not filtered_results:
        # If no images meet the threshold, still return at least the top 3
        return results[:3]
    
    return filtered_results

def analyze_image_with_model(image, query, model_info):
    """Analyze a single image with a specific model"""
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=model_info["temp"], 
            model_name=model_info["name"]
        )
        
        filename = image['filename']
        ocr_text = image['ocr_text']
        has_exact_match = image['has_exact_match']
        
        prompt = f"""
        Task: Evaluate the relevance of an image to a query based on its filename and extracted text content.

        Image filename: "{filename}"
        Query: "{query}"
        
        Extracted text from image (via OCR):
        {ocr_text}
        
        {'NOTE: The exact query phrase was found in the image text.' if has_exact_match else ''}

        Context: This image might be related to software development and deployment processes,
        specifically the process of committing and deploying a component in a system called Symbox.

        Instructions:
        1. Analyze how relevant the image appears to be to the query based on filename and text content
        2. Consider technical terms that might relate to software deployment
        3. Rate the relevance on a scale from 1-10 (10 being extremely relevant)
        4. Provide a brief explanation for your rating

        Format your response exactly as follows:
        Rating: [1-10]
        Explanation: [your concise explanation in 1-3 sentences]
        """
        
        # Get response from Groq
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Parse rating from response
        try:
            rating_line = [line for line in response_text.split('\n') if line.strip().startswith('Rating:')][0]
            rating_str = rating_line.split(':')[1].strip()
            rating = int(rating_str.split()[0]) if rating_str[0].isdigit() else 0
        except (IndexError, ValueError):
            rating = 0
            
        # Extract explanation
        try:
            explanation_lines = []
            capture = False
            for line in response_text.split('\n'):
                if line.strip().startswith('Explanation:'):
                    capture = True
                    line = line.replace('Explanation:', '').strip()
                    if line:
                        explanation_lines.append(line)
                elif capture:
                    explanation_lines.append(line.strip())
            explanation = ' '.join(explanation_lines)
        except:
            explanation = "Could not parse explanation"
            
        # Return result
        return {
            'model': model_info["name"],
            'model_description': model_info["description"],
            'rating': rating,
            'explanation': explanation
        }
    except Exception as e:
        return {
            'model': model_info["name"],
            'model_description': model_info["description"],
            'rating': 0,
            'explanation': f"Error: {str(e)}"
        }

def analyze_images_with_multiple_models(images, query, models):
    """Use multiple Groq LLMs to analyze the relevance of selected images to the query"""
    print(f"Analyzing {len(images)} images with {len(models)} Groq models...")
    
    # Process each image with all models
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}: {img['filename']}")
        
        # Use ThreadPoolExecutor to run model evaluations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), 2)) as executor:
            future_to_model = {
                executor.submit(analyze_image_with_model, img, query, model_info): model_info
                for model_info in models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_info = future_to_model[future]
                try:
                    result = future.result()
                    img['models_evaluation'].append(result)
                    print(f"  - {model_info['description']} evaluation complete")
                except Exception as exc:
                    print(f"  - {model_info['description']} generated an exception: {exc}")
                    img['models_evaluation'].append({
                        'model': model_info["name"],
                        'model_description': model_info["description"],
                        'rating': 0,
                        'explanation': f"Error: {str(exc)}"
                    })
        
        # Calculate average rating across all models
        working_evaluations = [eval_result for eval_result in img['models_evaluation'] 
                             if eval_result['rating'] > 0]
        if working_evaluations:
            total_rating = sum(eval_result['rating'] for eval_result in working_evaluations)
            img['avg_rating'] = total_rating / len(working_evaluations)
        else:
            img['avg_rating'] = 0
    
    # Re-sort images based on a combined score (similarity + average rating)
    for img in images:
        # Weighted score: 40% similarity, 60% average rating
        # Boost exact matches further
        exact_match_boost = 0.2 if img['has_exact_match'] else 0
        img['combined_score'] = (img['similarity'] * 0.4) + (img['avg_rating'] / 10 * 0.6) + exact_match_boost
    
    images.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return images

def generate_evaluation_matrix(results, models_used, execution_time):
    """Generate an evaluation matrix for the image search results"""
    if not results:
        return {
            "summary": "No images found with sufficient similarity to the query.",
            "details": "No data available.",
            "model_comparison": "No data available."
        }
        
    # Basic statistics
    total_images = len(results)
    avg_similarity = sum(r['similarity'] for r in results) / total_images if total_images > 0 else 0
    avg_ai_rating = sum(r['avg_rating'] for r in results) / total_images if total_images > 0 else 0
    exact_match_count = sum(1 for r in results if r['has_exact_match'])
    
    # Model performance metrics
    model_metrics = {}
    for model_info in models_used:
        model_name = model_info["description"]
        model_ratings = []
        for img in results:
            for eval_result in img['models_evaluation']:
                if eval_result['model_description'] == model_name and eval_result['rating'] > 0:
                    model_ratings.append(eval_result['rating'])
                    break
        
        if model_ratings:
            model_metrics[model_name] = {
                'avg_rating': sum(model_ratings) / len(model_ratings),
                'min_rating': min(model_ratings),
                'max_rating': max(model_ratings)
            }
    
    # Prepare matrix data
    matrix_data = [
        ["Total Images Found", total_images],
        ["Images with Exact Query Match", exact_match_count],
        ["Average Similarity Score", f"{avg_similarity:.4f}"],
        ["Average AI Relevance Rating", f"{avg_ai_rating:.2f}/10"],
        ["Highest Similarity Score", f"{results[0]['similarity']:.4f}" if results else "N/A"],
        ["Lowest Similarity Score", f"{results[-1]['similarity']:.4f}" if results else "N/A"],
        ["Processing Time", f"{execution_time:.2f} seconds"]
    ]
    
    # Add model metrics to matrix
    for model_name, metrics in model_metrics.items():
        matrix_data.append([
            f"{model_name} Avg Rating", 
            f"{metrics['avg_rating']:.2f}/10"
        ])
    
    # Prepare detailed results table
    details_data = []
    for i, r in enumerate(results, 1):
        model_ratings = []
        for model_info in models_used:
            model_name = model_info["description"]
            for eval_result in r['models_evaluation']:
                if eval_result['model_description'] == model_name:
                    if 'Error' in eval_result['explanation']:
                        model_ratings.append("Error")
                    else:
                        model_ratings.append(f"{eval_result['rating']}/10")
                    break
            else:
                model_ratings.append("N/A")
        
        # Add an indicator for exact matches
        exact_match_indicator = " [EXACT MATCH]" if r['has_exact_match'] else ""
        
        details_data.append([
            i,
            r['filename'] + exact_match_indicator,
            f"{r['similarity']:.4f}",
            f"{r['avg_rating']:.2f}/10",
            f"{r['combined_score']:.4f}",
            *model_ratings
        ])
    
    # Create headers for the details table
    details_headers = ["#", "Image", "Similarity", "Avg Rating", "Combined Score"]
    details_headers.extend([model_info["description"] for model_info in models_used])
    
    # Model comparison data
    model_comparison = []
    for img in results[:min(5, len(results))]:  # Top 5 images or less if fewer available
        # Add an indicator for exact matches
        exact_match_indicator = " [EXACT MATCH]" if img['has_exact_match'] else ""
        
        img_row = [img['filename'] + exact_match_indicator]
        for model_info in models_used:
            model_name = model_info["description"]
            for eval_result in img['models_evaluation']:
                if eval_result['model_description'] == model_name:
                    if 'Error' in eval_result['explanation']:
                        img_row.append("Error")
                    else:
                        img_row.append(f"{eval_result['rating']}/10")
                    break
            else:
                img_row.append("N/A")
        model_comparison.append(img_row)
    
    model_comparison_headers = ["Image"] + [model_info["description"] for model_info in models_used]
    
    return {
        "summary": tabulate(matrix_data, headers=["Metric", "Value"], tablefmt="grid"),
        "details": tabulate(details_data, headers=details_headers, tablefmt="grid"),
        "model_comparison": tabulate(model_comparison, headers=model_comparison_headers, tablefmt="grid")
    }

def generate_model_performance_comparison(results, models_used):
    """Generate a comparison of model performance across all images"""
    if not results:
        return "No data available for model performance comparison."
        
    # Initialize dictionaries to store model performance data
    model_consistency = {}  # Measure of how consistent the model ratings are
    model_agreement = {}    # Measure of how much each model agrees with others
    
    for model_info in models_used:
        model_name = model_info["description"]
        # Initialize model data
        model_consistency[model_name] = {
            'ratings': [],
            'std_dev': 0  # Standard deviation of ratings (lower means more consistent)
        }
        model_agreement[model_name] = {
            'agreements': []  # Will store differences with other models' ratings
        }
    
    # Collect all ratings by model
    for img in results:
        img_ratings = {}
        for eval_result in img['models_evaluation']:
            model_name = eval_result['model_description']
            rating = eval_result['rating']
            if rating > 0:  # Only consider valid ratings
                img_ratings[model_name] = rating
                model_consistency[model_name]['ratings'].append(rating)
        
        # Calculate agreement between models for this image
        for model1 in img_ratings:
            for model2 in img_ratings:
                if model1 != model2:
                    # Add the absolute difference to the agreements list
                    difference = abs(img_ratings[model1] - img_ratings[model2])
                    model_agreement[model1]['agreements'].append(difference)
    
    # Calculate standard deviation for each model's ratings
    for model_name in model_consistency:
        ratings = model_consistency[model_name]['ratings']
        if ratings:
            mean = sum(ratings) / len(ratings)
            variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
            std_dev = variance ** 0.5
            model_consistency[model_name]['std_dev'] = std_dev
    
    # Calculate average agreement score for each model
    for model_name in model_agreement:
        agreements = model_agreement[model_name]['agreements']
        if agreements:
            # Lower average means better agreement with other models
            model_agreement[model_name]['avg_difference'] = sum(agreements) / len(agreements)
    
    # Prepare comparison data
    comparison_data = []
    for model_info in models_used:
        model_name = model_info["description"]
        model_data = model_consistency[model_name]
        agreement_data = model_agreement[model_name]
        
        if model_data['ratings']:
            avg_rating = sum(model_data['ratings']) / len(model_data['ratings'])
            comparison_data.append([
                model_name,
                f"{avg_rating:.2f}/10",
                f"{model_data['std_dev']:.2f}",
                f"{agreement_data.get('avg_difference', 0):.2f}"
            ])
    
    # Sort by average rating descending
    comparison_data.sort(key=lambda x: float(x[1].split('/')[0]), reverse=True)
    
    if not comparison_data:
        return "No valid model performance data available."
        
    return tabulate(
        comparison_data, 
        headers=["Model", "Avg Rating", "Consistency (σ)", "Avg Difference"], 
        tablefmt="grid"
    )

if __name__ == "__main__":
    # Define paths
    image_folder = "images"
    
    print(f"Searching for images matching: \"{QUERY}\"")
    print(f"Looking in folder: {image_folder}")
    print(f"Using {len(GROQ_MODELS)} Groq models for analysis:")
    for model in GROQ_MODELS:
        print(f"  - {model['description']} ({model['name']})")
    
    # Start timing
    start_time = time.time()
    
    # Find matching images
    matching_images = find_matching_images(QUERY, image_folder)
    
    if not matching_images:
        print("No matching images found.")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Processing completed in {execution_time:.2f} seconds.")
        exit()
        
    # Analyze images with multiple Groq models
    print(f"Found {len(matching_images)} potentially matching images. Analyzing with Groq models...")
    analyzed_results = analyze_images_with_multiple_models(matching_images, QUERY, GROQ_MODELS)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Generate evaluation matrix
    evaluation_matrix = generate_evaluation_matrix(analyzed_results, GROQ_MODELS, execution_time)
    
    # Generate model performance comparison
    model_performance = generate_model_performance_comparison(analyzed_results, GROQ_MODELS)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(evaluation_matrix["summary"])
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(model_performance)
    
    print("\n" + "="*80)
    print("MODEL RATINGS COMPARISON (TOP IMAGES)")
    print("="*80)
    print(evaluation_matrix["model_comparison"])
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(evaluation_matrix["details"])
    
    print("\n" + "="*80)
    print("TOP 3 IMAGES WITH EXPLANATIONS")
    print("="*80)
    for i, img in enumerate(analyzed_results[:min(3, len(analyzed_results))], 1):
        print(f"{i}. {img['filename']}")
        print(f"   Similarity Score: {img['similarity']:.4f}")
        print(f"   Average AI Rating: {img['avg_rating']:.2f}/10")
        print(f"   Combined Score: {img['combined_score']:.4f}")
        print(f"   Contains Exact Match: {'Yes' if img['has_exact_match'] else 'No'}")
        print(f"   OCR Text Preview: {img['ocr_text'][:200]}...")
        print("   Model Evaluations:")
        
        # Sort evaluations by rating (highest first)
        sorted_evals = sorted(img['models_evaluation'], key=lambda x: x['rating'], reverse=True)
        
        for eval_result in sorted_evals:
            if 'Error' in eval_result['explanation']:
                print(f"     - {eval_result['model_description']}: Error")
                print(f"       {eval_result['explanation']}")
            else:
                print(f"     - {eval_result['model_description']}: {eval_result['rating']}/10")
                print(f"       {eval_result['explanation']}")
        print()
