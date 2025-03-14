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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This extension focuses on better handling abstract prompts
# It should be used with the main BLIP-2 contextual evaluation script

# Function to classify prompt abstractness
def classify_prompt_abstractness(prompt):
    """
    Classify a prompt on a scale from concrete to abstract
    Returns a score from 0.0 (very concrete) to 1.0 (very abstract)
    """
    # Keywords that indicate abstraction
    abstract_keywords = [
        'concept', 'abstract', 'representation', 'metaphor', 'symbolic', 
        'depicting', 'representing', 'illustrating', 'conceptual', 'idea',
        'notion', 'impression', 'theme', 'mood', 'feeling', 'aesthetic',
        'scene', 'scenario', 'setting', 'atmosphere', 'ambiance',
        'style', 'artistic', 'creative', 'imagination', 'perspective',
        'interpretation', 'visualization', 'portrayal', 'expression',
        'diagram', 'visual', 'image', 'picture', 'portrayal'
    ]
    
    # Keywords that indicate concreteness
    concrete_keywords = [
        'specific', 'exact', 'precise', 'detailed', 'literal', 
        'actual', 'real', 'physical', 'tangible', 'definite',
        'particular', 'clear', 'well-defined', 'explicit'
    ]
    
    # Count keyword occurrences
    abstract_count = sum(1 for word in abstract_keywords if word.lower() in prompt.lower())
    concrete_count = sum(1 for word in concrete_keywords if word.lower() in prompt.lower())
    
    # Calculate abstractness ratio
    total = abstract_count + concrete_count
    if total == 0:
        # No clear indicators, apply secondary heuristics
        
        # Check for specificity markers (numbers, measurements, colors)
        specificity_markers = re.findall(r'\d+|cm|mm|inch|foot|feet|meter|gram|pound|red|blue|green|yellow|black|white', prompt.lower())
        
        # Check for indefinite articles and qualifiers
        indefinite_markers = re.findall(r'\ba\b|\ban\b|\bsome\b|\bmany\b|\bvarious\b|\bseveral\b', prompt.lower())
        
        # Check sentence structure complexity
        sentences = sent_tokenize(prompt)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Combine these secondary factors
        abstract_score = 0.5  # Start neutral
        abstract_score -= len(specificity_markers) * 0.05  # More specific = less abstract
        abstract_score += len(indefinite_markers) * 0.05  # More indefinite = more abstract
        abstract_score += (avg_sentence_length - 10) * 0.01  # Longer sentences tend to be more abstract
        
        # Bound the score
        return max(0.0, min(1.0, abstract_score))
    else:
        return abstract_count / total

# Function to expand abstract concepts into concrete visual elements
def expand_abstract_concepts(prompt, abstractness_score):
    """
    Expand abstract concepts from the prompt into more concrete visual elements to check for
    """
    # Only apply expansion for more abstract prompts
    if abstractness_score < 0.6:
        return []
    
    # Abstract concept mappings to visual elements
    concept_mappings = {
        "event": ["people gathering", "decorations", "activity", "movement", "interaction", "crowded scene"],
        "significant activity": ["people performing tasks", "tools being used", "visible progress", "organized arrangement"],
        "place": ["location", "setting", "environment", "background", "landscape", "interior space"],
        "scene": ["setting", "environment", "composition", "arrangement", "background elements"],
        "diagram": ["lines", "shapes", "labels", "arrows", "organized layout", "structured information"],
        "concept": ["visual metaphor", "symbolic elements", "representative imagery", "illustrative components"],
        "abstract": ["geometric shapes", "non-representational elements", "symbolic imagery", "patterns"],
        "style": ["consistent visual treatment", "artistic technique", "color palette", "recognizable aesthetic"],
        "mood": ["lighting effects", "color tone", "composition", "emotional elements", "atmospheric qualities"],
        "representation": ["recognizable elements", "symbolic imagery", "visual metaphors", "illustrative components"]
    }
    
    expanded_elements = []
    
    # Look for abstract concepts in the prompt
    for concept, visual_elements in concept_mappings.items():
        if concept.lower() in prompt.lower():
            # Add the visual elements for this concept
            expanded_elements.extend(visual_elements)
    
    # If no specific concepts matched, add general abstract visual elements
    if not expanded_elements:
        expanded_elements = [
            "visual composition", "organized elements", "visual hierarchy", 
            "focal point", "visual balance", "meaningful arrangement",
            "purposeful design", "intentional layout", "thematic elements"
        ]
    
    return expanded_elements

# Function to generate additional abstract-aware questions
def generate_abstract_questions(prompt, abstractness_score, expanded_elements):
    """Generate questions that better handle abstract prompts"""
    
    questions = []
    
    # Only generate these specialized questions for more abstract prompts
    if abstractness_score < 0.5:
        return questions
    
    # General abstract evaluation questions
    abstract_questions = [
        "What is the overall theme or concept shown in this image?",
        "What visual elements in this image convey meaning or significance?",
        "How does the composition of this image communicate its purpose?",
        "What mood or atmosphere is created by this image?",
        "What visual metaphors or symbols can you identify in this image?"
    ]
    
    # Add general abstract questions
    questions.extend(abstract_questions)
    
    # Add questions about expanded concrete elements
    for element in expanded_elements[:3]:  # Limit to top 3 to avoid too many questions
        questions.append(f"Does this image contain {element}?")
        questions.append(f"How does {element} contribute to the overall meaning of this image?")
    
    return questions[:5]  # Limit to 5 questions

# Function to analyze BLIP-2 answers with abstractness awareness
def analyze_abstract_answers(answers, prompt, abstractness_score, expanded_elements):
    """Analyze BLIP-2 answers with special consideration for abstract prompts"""
    
    # Initialize metrics
    metrics = {
        "thematic_alignment": 0,
        "visual_element_presence": 0,
        "mood_consistency": 0,
        "conceptual_coherence": 0
    }
    
    # Compile all answer text
    all_answer_text = " ".join([answer.lower() for answer in answers.values()])
    
    # For abstract prompts, focus more on thematic alignment than literal matching
    if abstractness_score > 0.7:
        # Check for expanded concrete elements
        elements_found = sum(1 for element in expanded_elements if element.lower() in all_answer_text)
        if expanded_elements:
            metrics["visual_element_presence"] = elements_found / len(expanded_elements)
        
        # Check for thematic words from prompt
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        thematic_matches = sum(1 for word in prompt_words if word.lower() in all_answer_text)
        if prompt_words:
            metrics["thematic_alignment"] = thematic_matches / len(prompt_words)
        
        # Use text similarity for overall conceptual coherence
        vectorizer = TfidfVectorizer().fit_transform([prompt, all_answer_text])
        cosine_sim = cosine_similarity(vectorizer)[0, 1]
        metrics["conceptual_coherence"] = cosine_sim
    
    # Calculate the abstract-aware hallucination score
    # For abstract prompts, we weight conceptual alignment more heavily
    if abstractness_score > 0.7:
        hallucination_score = (
            metrics["thematic_alignment"] * 0.3 +
            metrics["visual_element_presence"] * 0.4 +
            metrics["conceptual_coherence"] * 0.3
        )
    else:
        # For more concrete prompts, use the regular analysis
        hallucination_score = None  # Will use the standard analysis
    
    metrics["abstract_hallucination_score"] = hallucination_score
    
    return metrics

# Function to adjust hallucination threshold based on abstractness
def adjust_hallucination_threshold(abstractness_score):
    """Return an adjusted hallucination threshold based on abstractness score"""
    # For very abstract prompts, we should be more lenient
    # For very concrete prompts, we should be more strict
    base_threshold = 0.5  # Standard threshold
    
    # Adjust threshold based on abstractness
    if abstractness_score > 0.7:
        # More lenient for highly abstract prompts
        return base_threshold - 0.15
    elif abstractness_score > 0.5:
        # Slightly more lenient for moderately abstract prompts
        return base_threshold - 0.05
    elif abstractness_score < 0.3:
        # More strict for very concrete prompts
        return base_threshold + 0.05
    else:
        # Use standard threshold for middle range
        return base_threshold

# Usage example (when incorporating into the main script):
"""
# Add to the main script before processing images
# For each prompt, first evaluate abstractness
abstractness_score = classify_prompt_abstractness(prompt)

# Generate additional questions based on abstractness
expanded_elements = expand_abstract_concepts(prompt, abstractness_score)
abstract_questions = generate_abstract_questions(prompt, abstractness_score, expanded_elements)

# Add these questions to your regular questions
all_questions = regular_questions + abstract_questions

# After getting answers, run both analyses
regular_metrics = analyze_answers(answers, prompt)
abstract_metrics = analyze_abstract_answers(answers, prompt, abstractness_score, expanded_elements)

# Decide which metrics to use based on abstractness
if abstractness_score > 0.7 and abstract_metrics["abstract_hallucination_score"] is not None:
    # Use abstract metrics for abstract prompts
    hallucination_score = abstract_metrics["abstract_hallucination_score"]
else:
    # Use regular metrics for concrete prompts
    hallucination_score = regular_metrics["hallucination_score"]

# Adjust threshold based on abstractness
threshold = adjust_hallucination_threshold(abstractness_score)
is_hallucination = hallucination_score < threshold

# Add abstractness info to results
result['abstractness_score'] = abstractness_score
result['adjusted_threshold'] = threshold
"""