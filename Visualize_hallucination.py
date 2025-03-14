import pandas as pd
import os
from PIL import Image

def loadTestData(directory):
    # Load the annotations CSV file
    annotations_df = pd.read_csv(directory + '/annotations.csv')

    # Load the 3k prompts CSV file
    prompts_df = pd.read_csv(directory + '/3k_prompts.csv')[:400]

    # Load the images
    images = []
    for filename in os.listdir(directory + '/3k-images'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory + '/3k-images', filename))
            images.append(img)
    return annotations_df, prompts_df, images

# Load the annotations CSV file
annotations_df = pd.read_csv('Hallucination/annotations.csv')

# Load the 3k prompts CSV file
prompts_df = pd.read_csv('Hallucination/3k_prompts.csv')

# Filter rows where quality is "not_realistic" in annotations_df
not_realistic_annotations = annotations_df[annotations_df['quality'] == 'not_realistic']

print(len(annotations_df))
print(len(prompts_df))
# Filter rows where quality is "not_realistic" in prompts_df
#not_realistic_prompts = prompts_df[annotations_df['quality'] == 'not_realistic']

# Display the filtered rows
print("Annotations with quality 'not_realistic':")
print(not_realistic_annotations)

#print("\nPrompts with quality 'not_realistic':")
#print(not_realistic_prompts)