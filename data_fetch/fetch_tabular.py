from datasets import load_dataset
import sys
import os
import json

# Add the root project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def fetch_tabular_data():
    # Load the dataset
    dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual")
    # Get the training split
    train_data = dataset['train']

    def filter_pii_and_language(example):
        # Check if language is English
        is_english = example['language'] == 'English'

        # Parse the JSON string to a list of dictionaries
        try:
            pii_spans = json.loads(example['pii_spans'])
            # Check if pii_spans contains any of the desired labels
            has_desired_pii = any(
                span['label'] in ['street_address', 'company', 'name']
                for span in pii_spans
            )
        except (json.JSONDecodeError, TypeError):
            has_desired_pii = False

        return is_english and has_desired_pii

    # Apply both filters
    filtered_data = train_data.filter(filter_pii_and_language)

    return filtered_data


if __name__ == "__main__":
    data = fetch_tabular_data()

    # Convert to list for proper iteration
    examples = list(data)

    # Print first 5 examples with their PII spans
    for i in range(min(5, len(examples))):
        example = examples[i]
        print("\nGenerated Text:", example['generated_text'])
        pii_spans = json.loads(example['pii_spans'])
        relevant_spans = [span for span in pii_spans
                          if span['label'] in ['street_address', 'company', 'name']]
        print("PII Spans:", relevant_spans)
        print("Language:", example['language'])
        print("-" * 80)