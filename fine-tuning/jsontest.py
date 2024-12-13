import json
import jsonschema
from typing import List, Dict

def validate_training_data(file_path: str) -> tuple[bool, List[str]]:
    """
    Validate the formatted training data for Ollama.
    Returns (is_valid, list_of_errors)
    """
    # Define the expected schema
    schema = {
        "type": "object",
        "required": ["prompt", "response"],
        "properties": {
            "prompt": {"type": "string", "minLength": 10},
            "response": {"type": "string", "minLength": 5}
        }
    }

    errors = []
    line_number = 0

    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    entry = json.loads(line.strip())
                    
                    # Validate against schema
                    jsonschema.validate(instance=entry, schema=schema)
                    
                    # Additional validations
                    if not entry['prompt'].startswith("### Human:"):
                        errors.append(f"Line {line_number}: Prompt should start with '### Human:'")
                    
                    if "### Assistant:" not in entry['prompt']:
                        errors.append(f"Line {line_number}: Prompt should contain '### Assistant:'")
                    
                    if len(entry['response']) > 2000:  # Example length check
                        errors.append(f"Line {line_number}: Response too long ({len(entry['response'])} chars)")
                    
                except json.JSONDecodeError:
                    errors.append(f"Line {line_number}: Invalid JSON format")
                except jsonschema.exceptions.ValidationError as e:
                    errors.append(f"Line {line_number}: {str(e)}")

    except FileNotFoundError:
        return False, ["Training file not found"]

    return len(errors) == 0, errors

def print_data_statistics(file_path: str) -> Dict:
    """
    Print statistics about the training data
    """
    stats = {
        "total_entries": 0,
        "avg_prompt_length": 0,
        "avg_response_length": 0,
        "shortest_response": float('inf'),
        "longest_response": 0
    }

    try:
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                stats['total_entries'] += 1
                stats['avg_prompt_length'] += len(entry['prompt'])
                stats['avg_response_length'] += len(entry['response'])
                stats['shortest_response'] = min(stats['shortest_response'], len(entry['response']))
                stats['longest_response'] = max(stats['longest_response'], len(entry['response']))

        if stats['total_entries'] > 0:
            stats['avg_prompt_length'] /= stats['total_entries']
            stats['avg_response_length'] /= stats['total_entries']

    except FileNotFoundError:
        print("Training file not found")
        return stats

    return stats

# Usage example
if __name__ == "__main__":
    file_path = 'nutrisage_training.jsonl'
    
    print("Validating training data...")
    is_valid, errors = validate_training_data(file_path)
    
    if is_valid:
        print("✅ Training data format is valid!")
        
        # Print statistics
        stats = print_data_statistics(file_path)
        print("\nData Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Average prompt length: {stats['avg_prompt_length']:.1f} characters")
        print(f"Average response length: {stats['avg_response_length']:.1f} characters")
        print(f"Shortest response: {stats['shortest_response']} characters")
        print(f"Longest response: {stats['longest_response']} characters")
    else:
        print("❌ Found validation errors:")
        for error in errors:
            print(f"  - {error}")