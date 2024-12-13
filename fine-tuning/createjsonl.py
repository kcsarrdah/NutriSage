import json
import random

# Function to load and combine all our Q&A pairs
def combine_training_data():
    categories = [
        "nutritional_values",
        "dietary_guidelines",
        "health_impact",
        "food_comparisons",
        "portion_control",
        "special_diets",
        "meal_planning",
        "recipe_modifications",
        "common_misconceptions",
        "edge_cases"
    ]
    
    # Load and combine all data
    all_qa_pairs = []
    
    for category in categories:
        with open(f'{category}_qa.json', 'r') as f:
            data = json.load(f)
            qa_pairs = data['training_data'][category]
            all_qa_pairs.extend(qa_pairs)
            
    return all_qa_pairs

def format_for_ollama(qa_pairs):
    """Convert Q&A pairs to Ollama format"""
    formatted_data = []
    
    for pair in qa_pairs:
        # Create the prompt template
        formatted_prompt = f"""### Human: {pair['question']}

### Assistant: Let me provide accurate nutritional information based on reliable sources.

{pair['answer']}

This information comes from {pair['source']}.
"""
        # Add to formatted data
        formatted_data.append({
            "prompt": formatted_prompt,
            "response": pair['answer']
        })
    
    return formatted_data

# Save in Ollama format
def save_formatted_data(formatted_data, output_file):
    with open(output_file, 'w') as f:
        for item in formatted_data:
            # Ollama expects JSONL format (one JSON object per line)
            f.write(json.dumps(item) + '\n')

# Main process
if __name__ == "__main__":
    # Load all Q&A pairs
    qa_pairs = combine_training_data()  # from previous script
    
    # Format for Ollama
    formatted_data = format_for_ollama(qa_pairs)
    
    # Save to JSONL file
    save_formatted_data(formatted_data, 'nutrisage_training.jsonl')