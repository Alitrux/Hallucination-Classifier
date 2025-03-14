import torch
import clip
import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
print("Model loaded")

# Load prompts data
prompts_file = "Hallucination/3k_prompts.csv"
df = pd.read_csv(prompts_file)
print(f"Loaded {len(df)} prompt-image pairs")

# Check the column names to ensure we're using the right ones
print(f"DataFrame columns: {df.columns.tolist()}")

# Assume the image and prompt columns are named 'image' and 'prompt'
# Change these to match your actual column names
IMAGE_COL = 'image'  # Update this if necessary
PROMPT_COL = 'prompt'  # Update this if necessary

# Base directory for images
image_dir = "Hallucination/3k-images/"

def extract_key_elements(prompt):
    """Extract key elements from the prompt"""
    tokens = word_tokenize(prompt)
    tagged = pos_tag(tokens)
    
    # Extract nouns and adjectives (key objects and their attributes)
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
    
    # Extract named entities
    named_entities = []
    chunked = ne_chunk(tagged)
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            named_entities.append(entity)
    
    return {
        'nouns': nouns,
        'adjectives': adjectives,
        'named_entities': named_entities
    }

def calculate_clip_similarity(image_tensor, text, model, device):
    """Calculate CLIP similarity between image and text"""
    text_input = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_input)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine similarity)
        similarity = (100.0 * image_features @ text_features.T).item()
    
    return similarity

# Process each image-prompt pair
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    image_name = row[IMAGE_COL]
    prompt = row[PROMPT_COL]
    
    # Construct the full image path
    image_path = os.path.join(image_dir, image_name)
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Calculate overall similarity
        overall_similarity = calculate_clip_similarity(image_input, prompt, model, device)
        
        # Extract key elements from the prompt
        key_elements = extract_key_elements(prompt)
        
        # Calculate element-specific similarities
        element_similarities = {}
        
        # Test each noun individually
        for noun in key_elements['nouns']:
            element_similarities[f'noun_{noun}'] = calculate_clip_similarity(image_input, noun, model, device)
        
        # Test adjective-noun combinations
        for adj in key_elements['adjectives']:
            for noun in key_elements['nouns']:
                combo = f"{adj} {noun}"
                element_similarities[f'combo_{combo}'] = calculate_clip_similarity(image_input, combo, model, device)
        
        # Test named entities
        for entity in key_elements['named_entities']:
            element_similarities[f'entity_{entity}'] = calculate_clip_similarity(image_input, entity, model, device)
        
        # Calculate semantic element coverage
        if element_similarities:
            element_avg = np.mean(list(element_similarities.values()))
            element_min = np.min(list(element_similarities.values()))
        else:
            element_avg = overall_similarity
            element_min = overall_similarity
        
        # Store results
        result = {
            'image': image_name,
            'prompt': prompt,
            'overall_similarity': overall_similarity,
            'element_avg_similarity': element_avg,
            'element_min_similarity': element_min,
            'nouns': ', '.join(key_elements['nouns']),
            'adjectives': ', '.join(key_elements['adjectives']),
            'named_entities': ', '.join(key_elements['named_entities'])
        }
        
        # Add individual element similarities
        result.update(element_similarities)
        
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save detailed results
results_df.to_csv("detailed_clip_similarity_results.csv", index=False)

# Analyze results
print("\nAnalysis of similarity scores:")
print(f"Overall similarity - Mean: {results_df['overall_similarity'].mean():.2f}, Median: {results_df['overall_similarity'].median():.2f}")
print(f"Element average similarity - Mean: {results_df['element_avg_similarity'].mean():.2f}, Median: {results_df['element_avg_similarity'].median():.2f}")
print(f"Element minimum similarity - Mean: {results_df['element_min_similarity'].mean():.2f}, Median: {results_df['element_min_similarity'].median():.2f}")

# Identify hallucinations using multiple criteria
# 1. Low overall similarity
low_overall = results_df['overall_similarity'] < results_df['overall_similarity'].quantile(0.25)

# 2. Low minimum element similarity (missing critical elements)
low_element_min = results_df['element_min_similarity'] < results_df['element_min_similarity'].quantile(0.25)

# 3. Gap between overall and element-level similarities
# (indicates the model might be "seeing" something different than what's described)
similarity_gap = (results_df['overall_similarity'] - results_df['element_avg_similarity']).abs()
high_gap = similarity_gap > similarity_gap.quantile(0.75)

# Combine criteria
potential_hallucinations = results_df[low_overall | low_element_min | high_gap].copy()

# Add reason for flagging
potential_hallucinations['flagged_reason'] = ''
potential_hallucinations.loc[low_overall, 'flagged_reason'] += 'Low overall similarity; '
potential_hallucinations.loc[low_element_min, 'flagged_reason'] += 'Missing critical elements; '
potential_hallucinations.loc[high_gap, 'flagged_reason'] += 'Discrepancy between overall and element similarities; '

# Save potential hallucinations
potential_hallucinations.to_csv("advanced_potential_hallucinations.csv", index=False)

print(f"\nFound {len(potential_hallucinations)} potential hallucinations out of {len(results_df)} images ({len(potential_hallucinations)/len(results_df)*100:.1f}%)")
print("Results saved to detailed_clip_similarity_results.csv and advanced_potential_hallucinations.csv")

# Generate summary statistics for hallucination types
hallucination_types = {
    'Low overall similarity': len(potential_hallucinations[low_overall]),
    'Missing critical elements': len(potential_hallucinations[low_element_min]),
    'Discrepancy in similarities': len(potential_hallucinations[high_gap])
}

print("\nHallucination types breakdown:")
for htype, count in hallucination_types.items():
    print(f"- {htype}: {count} images ({count/len(results_df)*100:.1f}%)")