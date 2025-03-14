import torch
import clip
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

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

# Create a results dataframe
results = []

# Process each image-prompt pair
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    image_name = row[IMAGE_COL]
    prompt = row[PROMPT_COL]
    
    # Construct the full image path
    image_path = os.path.join(image_dir, image_name)
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize the prompt
        text_input = clip.tokenize([prompt]).to(device)
        
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity (cosine similarity)
            similarity = (100.0 * image_features @ text_features.T).item()
        
        # Store results
        results.append({
            'image': image_name,
            'prompt': prompt,
            'similarity_score': similarity
        })
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv("Hallucination/clip_similarity_results.csv", index=False)

# Analyze results
print("\nAnalysis of similarity scores:")
print(f"Min similarity: {results_df['similarity_score'].min():.2f}")
print(f"Max similarity: {results_df['similarity_score'].max():.2f}")
print(f"Mean similarity: {results_df['similarity_score'].mean():.2f}")
print(f"Median similarity: {results_df['similarity_score'].median():.2f}")

# # Define potential hallucination thresholds
# low_threshold = results_df['similarity_score'].quantile(0.25)
# print(f"\nPotential hallucination threshold (25th percentile): {low_threshold:.2f}")

# # Identify potential hallucinations
# potential_hallucinations = results_df[results_df['similarity_score'] < low_threshold]
# print(f"Found {len(potential_hallucinations)} potential hallucinations out of {len(results_df)} images ({len(potential_hallucinations)/len(results_df)*100:.1f}%)")

# # Save potential hallucinations
# potential_hallucinations.to_csv("potential_hallucinations.csv", index=False)
# print("Results saved to clip_similarity_results.csv and potential_hallucinations.csv")