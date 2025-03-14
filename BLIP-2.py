import pandas as pd
import torch
from PIL import Image
import os
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import json
import re

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP-2 model
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
model.to(device)
print(f"Loaded BLIP-2 model: {model_name}")

# Load prompts data
prompts_file = "3k_prompts.csv"
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

# Function to generate contextual questions based on prompt
def generate_questions_for_prompt(prompt):
    """Generate a set of contextual questions based on the prompt"""
    
    # Basic question templates
    general_questions = [
        "What is shown in this image?",
        "Describe the main elements visible in this image.",
        "What is the primary subject of this image?",
        "Does this image match a description of '{}'?",
    ]
    
    # Generate prompt-specific questions
    specific_questions = []
    
    # Extract key phrases from the prompt
    sentences = sent_tokenize(prompt)
    
    for sentence in sentences:
        # Add a direct question about the sentence
        specific_questions.append(f"Does this image show {sentence.lower().rstrip('.')}?")
        
        # Look for key elements that might indicate abstraction
        abstract_indicators = ["showing", "depicting", "representing", "illustrating", 
                              "scene", "view", "image", "picture", "concept", "diagram"]
        
        for indicator in abstract_indicators:
            if indicator in sentence.lower():
                # Find what comes after the indicator
                pattern = f"{indicator} (of |a |an |the )?(.+?)(\\.|,|;|$)"
                matches = re.findall(pattern, sentence.lower())
                if matches:
                    for match in matches:
                        concept = match[1].strip()
                        if concept:
                            specific_questions.append(f"Is this a {indicator} of {concept}?")
                            specific_questions.append(f"What elements in this image represent {concept}?")
    
    # Compile all questions
    all_questions = general_questions + specific_questions
    
    # Add the prompt as a direct format to one of the general questions
    all_questions[3] = all_questions[3].format(prompt)
    
    # Limit to a reasonable number of questions (avoid too many API calls)
    return all_questions[:8]  # Limit to 8 questions per image

# Function to evaluate image against questions using BLIP-2
def evaluate_with_blip2(image_path, questions):
    """Use BLIP-2 to answer questions about the image"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        answers = {}
        for question in questions:
            # Process image and question
            inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
            
            # Generate answer
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=50)
                answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            answers[question] = answer
            
        return answers
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"error": str(e)}

# Function to analyze answers for hallucination indicators
def analyze_answers(answers, prompt):
    """Analyze BLIP-2 answers to determine hallucination likelihood"""
    
    # Initialize metrics
    metrics = {
        "positive_matches": 0,
        "negative_matches": 0,
        "confidence_score": 0,
        "answer_relevance": 0,
        "key_element_presence": 0,
        "contradiction_score": 0
    }
    
    # Count positive affirmations
    yes_patterns = ["yes", "correct", "true", "indeed", "shows", "depicts", "contains", "displays"]
    no_patterns = ["no", "not", "doesn't", "does not", "isn't", "is not", "cannot", "can't"]
    
    # Check yes/no questions
    yes_no_questions = [q for q in answers.keys() if q.startswith("Does") or q.startswith("Is")]
    for question in yes_no_questions:
        answer = answers[question].lower()
        
        # Check for positive matches
        if any(pattern in answer for pattern in yes_patterns):
            metrics["positive_matches"] += 1
        
        # Check for negative matches
        if any(pattern in answer for pattern in no_patterns):
            metrics["negative_matches"] += 1
    
    # Look for key elements from the prompt in the answers
    prompt_keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))  # Words with 4+ chars
    
    all_answers_text = " ".join([a.lower() for a in answers.values()])
    found_keywords = [keyword for keyword in prompt_keywords if keyword in all_answers_text]
    if prompt_keywords:
        metrics["key_element_presence"] = len(found_keywords) / len(prompt_keywords)
    
    # Check for contradictions between answers
    contradictions = 0
    answer_list = list(answers.values())
    for i in range(len(answer_list)):
        for j in range(i+1, len(answer_list)):
            # Simple contradiction detection
            if ("yes" in answer_list[i].lower() and "no" in answer_list[j].lower()) or \
               ("no" in answer_list[i].lower() and "yes" in answer_list[j].lower()):
                contradictions += 1
    
    if yes_no_questions:
        metrics["contradiction_score"] = contradictions / len(yes_no_questions)
    
    # Calculate overall confidence score
    total_questions = len(yes_no_questions)
    if total_questions > 0:
        metrics["confidence_score"] = metrics["positive_matches"] / total_questions
    
    # Calculate hallucination probability
    # Lower score means higher probability of hallucination
    if total_questions > 0:
        hallucination_score = (
            (metrics["positive_matches"] - metrics["negative_matches"]) / total_questions * 0.4 +
            metrics["key_element_presence"] * 0.4 - 
            metrics["contradiction_score"] * 0.2
        )
        hallucination_score = max(0, min(1, (hallucination_score + 1) / 2))  # Normalize to 0-1
    else:
        hallucination_score = 0.5  # Neutral if no yes/no questions
    
    metrics["hallucination_score"] = hallucination_score
    
    return metrics

# Process each image-prompt pair
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    image_name = row[IMAGE_COL]
    prompt = row[PROMPT_COL]
    
    # Construct the full image path
    image_path = os.path.join(image_dir, image_name)
    
    try:
        # Generate questions for this prompt
        questions = generate_questions_for_prompt(prompt)
        
        # Get answers from BLIP-2
        answers = evaluate_with_blip2(image_path, questions)
        
        # Analyze answers for hallucination indicators
        metrics = analyze_answers(answers, prompt)
        
        # Determine if this is a likely hallucination
        is_hallucination = metrics["hallucination_score"] < 0.5
        
        # Store results
        result = {
            'image': image_name,
            'prompt': prompt,
            'questions_answers': json.dumps(answers),
            'hallucination_score': metrics["hallucination_score"],
            'is_likely_hallucination': is_hallucination,
            'positive_matches': metrics["positive_matches"],
            'negative_matches': metrics["negative_matches"],
            'key_element_presence': metrics["key_element_presence"],
            'contradiction_score': metrics["contradiction_score"]
        }
        
        results.append(result)
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        results.append({
            'image': image_name,
            'prompt': prompt,
            'error': str(e)
        })
        
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save detailed results
results_df.to_csv("blip2_hallucination_results.csv", index=False)

# Analyze results
if 'hallucination_score' in results_df.columns:
    print("\nAnalysis of hallucination scores:")
    print(f"Mean hallucination score: {results_df['hallucination_score'].mean():.2f}")
    print(f"Median hallucination score: {results_df['hallucination_score'].median():.2f}")
    
    # Identify likely hallucinations
    hallucinations = results_df[results_df['is_likely_hallucination'] == True]
    print(f"\nFound {len(hallucinations)} likely hallucinations out of {len(results_df)} images ({len(hallucinations)/len(results_df)*100:.1f}%)")
    
    # Save likely hallucinations
    hallucinations.to_csv("blip2_likely_hallucinations.csv", index=False)

print("Results saved to blip2_hallucination_results.csv and blip2_likely_hallucinations.csv")

# Optional: Create a detailed report for each prompt with visualizations
if len(results_df) <= 20:  # Only for smaller datasets to avoid excessive processing
    print("Generating detailed reports for each prompt...")
    
    for idx, row in results_df.iterrows():
        if 'questions_answers' in row:
            qa_dict = json.loads(row['questions_answers'])
            
            print(f"\n--- Detailed Report for Image: {row['image']} ---")
            print(f"Prompt: {row['prompt']}")
            print(f"Hallucination Score: {row['hallucination_score']:.2f} ({'Likely hallucination' if row['is_likely_hallucination'] else 'Not a hallucination'})")
            print("\nQuestion-Answer Pairs:")
            
            for q, a in qa_dict.items():
                print(f"Q: {q}")
                print(f"A: {a}")
                print("-" * 50)
            
            print("=" * 80)