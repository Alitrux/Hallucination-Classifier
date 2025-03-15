import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load the results data
results_file = "detailed_clip_similarity_results.csv"
results_df = pd.read_csv(results_file)

print(f"Loaded {len(results_df)} results")

# Create output directory
os.makedirs("simple_viz", exist_ok=True)

# 1. Simple histogram of the three main similarity metrics
plt.figure(figsize=(10, 6))
plt.hist(results_df['overall_similarity'], alpha=0.7, bins=30, label='Overall Similarity')
plt.hist(results_df['element_avg_similarity'], alpha=0.7, bins=30, label='Avg Element Similarity')
plt.hist(results_df['element_min_similarity'], alpha=0.7, bins=30, label='Min Element Similarity')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Similarity Scores')
plt.legend()
plt.savefig('simple_viz/similarity_distribution.png')
plt.close()

# 2. Simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['overall_similarity'], results_df['element_min_similarity'], alpha=0.5)
plt.xlabel('Overall Similarity')
plt.ylabel('Minimum Element Similarity')
plt.title('Overall vs Minimum Element Similarity')
plt.grid(True)
plt.savefig('simple_viz/similarity_scatter.png')
plt.close()

# 3. Display the worst images (lowest minimum element similarity)
sorted_results = results_df.sort_values('element_min_similarity')
worst_images = sorted_results.head(9)  # Get the 9 worst images

# Create a figure with subplots for the worst images
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, (_, row) in enumerate(worst_images.iterrows()):
    if i >= 9:  # Limit to 9 images
        break
        
    # Get image path
    image_name = row['image']
    image_path = os.path.join("Hallucination/3k-images/", image_name)
    
    try:
        # Load and display image
        img = Image.open(image_path).convert('RGB')
        axes[i].imshow(img)
        
        # Add title with key information
        title = f"Image: {image_name}\n"
        title += f"Overall: {row['overall_similarity']:.1f}, "
        title += f"Min Element: {row['element_min_similarity']:.1f}\n"
        title += f"Prompt: {row['prompt'][:50]}..."  # Truncate long prompts
        
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    except Exception as e:
        print(f"Error loading image {image_name}: {e}")
        axes[i].text(0.5, 0.5, f"Error loading image: {image_name}", 
                    ha='center', va='center')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('simple_viz/worst_images.png', dpi=300)
plt.close()

# 4. Create a simple table of the worst 20 images
worst_20 = sorted_results.head(20)[['image', 'prompt', 'overall_similarity', 'element_min_similarity', 'element_avg_similarity']]
worst_20.to_csv('simple_viz/worst_20_images.csv', index=False)

print("Simple visualizations created in 'simple_viz' directory")