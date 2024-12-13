import json
import random

def format_for_ollama(data):
    """Convert Q&A pairs to Ollama format"""
    formatted_data = []
    
    # Process each category from our data
    for category in data["training_data"]:
        category_data = data["training_data"][category]
        for pair in category_data:
            formatted_prompt = f"""### Human: {pair['question']}

### Assistant: Let me provide accurate nutritional information based on reliable sources.

{pair['answer']}

This information comes from {pair['source']}.
"""
            formatted_data.append({
                "prompt": formatted_prompt,
                "response": pair['answer']
            })
    
    return formatted_data

def clean_data(formatted_data):
    """Clean and format the data"""
    cleaned_data = []
    removed_count = 0
    
    for item in formatted_data:
        # Clean whitespace and formatting
        item['prompt'] = ' '.join(item['prompt'].split())
        item['response'] = ' '.join(item['response'].split())
        
        # Validate entry lengths (minimum 10 characters)
        if len(item['prompt']) > 10 and len(item['response']) > 10:
            cleaned_data.append(item)
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} invalid entries during cleaning")
    return cleaned_data

def split_data(cleaned_data, train_ratio=0.8):
    """Split data into training and validation sets"""
    # Shuffle data
    random.shuffle(cleaned_data)
    
    # Calculate split point
    split_idx = int(len(cleaned_data) * train_ratio)
    
    # Split data
    train_data = cleaned_data[:split_idx]
    val_data = cleaned_data[split_idx:]
    
    print(f"Data split into {len(train_data)} training and {len(val_data)} validation examples")
    return train_data, val_data

def save_formatted_data(train_data, val_data):
    """Save the formatted data in JSONL format"""
    # Save training data
    with open('nutrisage_train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation data
    with open('nutrisage_val.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Successfully saved:")
    print(f"- {len(train_data)} training examples to nutrisage_train.jsonl")
    print(f"- {len(val_data)} validation examples to nutrisage_val.jsonl")

def process_training_data(input_file):
    """Main process to convert, clean, and split training data"""
    try:
        # Load the training data
        print("Loading training data...")
        with open(input_file, 'r') as f:
            training_data = json.load(f)
        
        # Format for Ollama
        print("\nFormatting data for Ollama...")
        formatted_data = format_for_ollama(training_data)
        print(f"Created {len(formatted_data)} formatted examples")
        
        # Clean data
        print("\nCleaning and validating data...")
        cleaned_data = clean_data(formatted_data)
        print(f"Retained {len(cleaned_data)} valid examples")
        
        # Split data
        print("\nSplitting into training and validation sets...")
        train_data, val_data = split_data(cleaned_data)
        
        # Save files
        print("\nSaving formatted data...")
        save_formatted_data(train_data, val_data)
        
        print("\nProcess completed successfully!")
        
    except FileNotFoundError:
        print("Error: Could not find the training data file")
    except json.JSONDecodeError:
        print("Error: The training data file is not valid JSON")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_file = "../data/training_data.json"
    process_training_data(input_file)