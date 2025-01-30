import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import numpy as np
from seqeval.metrics import classification_report
import ast
import time
from datetime import datetime
import warnings
import os
import json
import argparse

warnings.filterwarnings('ignore')


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomCallback(TrainerCallback):
    def __init__(self, save_path):
        self.training_start = None
        self.last_log = None
        self.best_f1 = 0.0
        self.save_path = save_path

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start = time.time()
        print("\nTraining Progress:")
        print("-" * 100)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_time = time.time()
        if self.training_start is not None:
            elapsed = current_time - self.training_start
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            progress = (state.global_step / state.max_steps) * 100

            print(f"\nStep: {state.global_step}/{state.max_steps} ({progress:.1f}%)")
            print(f"Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")

            if "loss" in logs:
                print(f"Training loss: {logs['loss']:.4f}")

            if "eval_macro_f1" in logs:
                print("\nEvaluation Metrics:")
                print(f"Macro F1: {logs['eval_macro_f1']:.4f}")

                #for entity in ["NAME", "COMPANY", "ADDRESS"]:
                for entity in ["name", "company", "street_address"]:
                    if f"eval_{entity}_f1" in logs:
                        print(f"{entity:8} - F1: {logs[f'eval_{entity}_f1']:.4f}, "
                              f"Precision: {logs[f'eval_{entity}_precision']:.4f}, "
                              f"Recall: {logs[f'eval_{entity}_recall']:.4f}")

                if logs["eval_macro_f1"] > self.best_f1:
                    self.best_f1 = logs["eval_macro_f1"]
                    print(f"\n*** New best macro F1: {self.best_f1:.4f} ***")


class NERPredictor:
    def __init__(self, model_name="xlm-roberta-base", labels=None, save_path="./ner_model"):
        if labels is None:
            self.labels = ["O", "name", "company", "street_address"]  # Changed to lowercase
        else:
            self.labels = labels

        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.model_name = model_name
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def save_model(self, save_path):
        """Save the model, tokenizer, and configuration"""
        os.makedirs(save_path, exist_ok=True)

        # Save model state dict separately
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model_state.bin'))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save model configuration
        self.model.config.save_pretrained(save_path)

        # Save additional configuration
        config = {
            'labels': self.labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'base_model_name': self.model_name,
            'model_type': 'xlm-roberta'
        }

        with open(os.path.join(save_path, 'ner_config.json'), 'w') as f:
            json.dump(config, f)

        print(f"\nModel saved to {save_path}")

    @classmethod
    def load_model(cls, model_path):
        """Load a saved model, tokenizer, and configuration"""
        # Load our custom configuration
        config_path = os.path.join(model_path, 'ner_config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"No configuration file found at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create instance with saved labels
        instance = cls(model_name=model_path, labels=config['labels'])

        # Load model configuration and weights
        instance.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(config['labels']),
            id2label=config['id2label'],
            label2id=config['label2id'],
            local_files_only=True
        )

        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Set model to evaluation mode
        instance.model.eval()

        print(f"\nModel loaded from {model_path}")
        return instance

    def predict(self, text):
        """Predict entities in the given text"""
        device = next(self.model.parameters()).device
        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            max_length=512
        )

        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

            predictions = predictions[0].cpu().numpy()
            offset_mapping = offset_mapping[0].numpy()

            entities = []
            current_entity = {"text": "", "label": None, "start": None}

            for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
                if offset[0] == 0 and offset[1] == 0:  # Skip special tokens
                    continue

                pred_label = self.id2label[pred]

                if pred_label != "O":
                    # Start of new entity or continuation
                    if current_entity["label"] is None:
                        current_entity = {
                            "text": text[offset[0]:offset[1]],
                            "label": pred_label,
                            "start": offset[0]
                        }
                    else:
                        # Append if same label and continuous
                        if pred_label == current_entity["label"]:
                            current_entity["text"] += text[offset[0]:offset[1]]
                        else:
                            # Save previous entity and start new one
                            if current_entity["text"].strip():
                                entities.append((current_entity["text"].strip(), current_entity["label"]))
                            current_entity = {
                                "text": text[offset[0]:offset[1]],
                                "label": pred_label,
                                "start": offset[0]
                            }
                elif current_entity["label"] is not None:
                    # End of current entity
                    if current_entity["text"].strip():
                        entities.append((current_entity["text"].strip(), current_entity["label"]))
                    current_entity = {"text": "", "label": None, "start": None}

            # Add final entity if exists
            if current_entity["label"] is not None and current_entity["text"].strip():
                entities.append((current_entity["text"].strip(), current_entity["label"]))

            return entities

    def compute_metrics(self, pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        # Debug: Print raw predictions shape and values
        print(f"\nDebug: Raw predictions shape: {predictions.shape}")
        print(f"Debug: Raw labels shape: {labels.shape}")

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Debug: Print sample of predictions vs actual
        print("\nDebug: Sample predictions vs actual (first 3 sequences):")
        for i in range(min(3, len(true_predictions))):
            print(f"\nSequence {i + 1}:")
            for pred, true in zip(true_predictions[i], true_labels[i]):
                if pred != "O" or true != "O":  # Only show non-O labels for clarity
                    print(f"Predicted: {pred:15} Actual: {true}")

        results = classification_report(true_labels, true_predictions, output_dict=True)

        # Debug: Print detailed metrics per label
        print("\nDebug: Detailed metrics per label:")
        for label in ["name", "company", "street_address"]:
            if label in results:
                print(f"\n{label}:")
                print(f"  Precision: {results[label]['precision']:.4f}")
                print(f"  Recall: {results[label]['recall']:.4f}")
                print(f"  F1: {results[label]['f1-score']:.4f}")
                print(f"  Support: {results[label]['support']}")

        metrics = {}
        for label in ["name", "company", "street_address"]:
            if label in results:
                metrics[f"{label}_precision"] = results[label]["precision"]
                metrics[f"{label}_recall"] = results[label]["recall"]
                metrics[f"{label}_f1"] = results[label]["f1-score"]

        metrics["macro_precision"] = results["macro avg"]["precision"]
        metrics["macro_recall"] = results["macro avg"]["recall"]
        metrics["macro_f1"] = results["macro avg"]["f1-score"]

        return metrics

    def encode_dataset(self, texts, labels):
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_offsets_mapping=True)
        encoded_labels = self._encode_labels(labels, encodings)
        return encodings, encoded_labels

    def _encode_labels(self, labels, encodings):
        """Encode labels for token classification"""
        encoded_labels = []

        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100

            # Handle special tokens first
            doc_enc_labels[0] = self.label2id["O"]  # [CLS] token
            doc_enc_labels[-1] = self.label2id["O"]  # [SEP] token

            # Sort spans by start position for proper handling of nested entities
            sorted_labels = sorted(doc_labels, key=lambda x: (x[0], -x[1]))

            # Process each token
            for i, (start, end) in enumerate(doc_offset[1:-1], start=1):
                if start == 0 and end == 0:  # Skip special tokens
                    continue

                # Default to O
                doc_enc_labels[i] = self.label2id["O"]

                # Find matching entity span
                for span_start, span_end, label in sorted_labels:
                    # Check if token falls within entity span
                    if start >= span_start and end <= span_end:
                        doc_enc_labels[i] = self.label2id[label]
                        break

            encoded_labels.append(doc_enc_labels.tolist())

            # Debug logging
            if len(encoded_labels) <= 3:  # Only show first 3 documents
                print(f"\nDocument {len(encoded_labels)} encoding:")
                print("Original spans:", [(s, e, l) for s, e, l in sorted_labels])
                for j, (token_start, token_end) in enumerate(doc_offset):
                    if token_start != 0 or token_end != 0:  # Show non-special tokens
                        token = self.tokenizer.decode([encodings.input_ids[len(encoded_labels) - 1][j]])
                        label = self.id2label[doc_enc_labels[j]]
                        if label != "O":
                            print(f"Token: '{token.strip()}' ({token_start}:{token_end}) → {label}")

        return encoded_labels

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              epochs=3, batch_size=8, learning_rate=2e-5, use_cpu=False):
        device = "cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        self.model.to(device)

        # Print sample of training data for verification
        print("\nSample of training data:")
        for i in range(min(3, len(train_texts))):
            print(f"\nText {i + 1}: {train_texts[i]}")
            print(f"Labels {i + 1}: {train_labels[i]}")

        train_encodings, train_enc_labels = self.encode_dataset(train_texts, train_labels)

        # Verify encoded labels
        print("\nVerifying encoded labels:")
        sample_idx = 0
        tokens = self.tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][sample_idx])
        enc_labels = train_enc_labels[sample_idx]
        print("\nTokens and their encoded labels:")
        for token, label in zip(tokens, enc_labels):
            if label != -100:
                print(f"{token}: {self.id2label[label]}")

        train_dataset = NERDataset(train_encodings, train_enc_labels)

        if val_texts and val_labels:
            val_encodings, val_enc_labels = self.encode_dataset(val_texts, val_labels)
            val_dataset = NERDataset(val_encodings, val_enc_labels)
        else:
            val_dataset = None

        training_args = TrainingArguments(
            output_dir="./ner_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_dir="./logs",
            logging_steps=50,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,  # Add warmup
            weight_decay=0.01,  # Add weight decay
            fp16=True if device == "cuda" else False,
            dataloader_num_workers=0,
            optim="adamw_torch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[CustomCallback(self.save_path)]
        )

        print("\nTraining started:")
        print(f"Number of training examples: {len(train_dataset)}")
        if val_dataset:
            print(f"Number of validation examples: {len(val_dataset)}")
        print(f"Number of epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print("=" * 50 + "\n")

        train_result = trainer.train()

        # Save the final model
        trainer.save_model(self.save_path)

        # Save tokenizer and configuration
        self.tokenizer.save_pretrained(self.save_path)
        self.model.config.save_pretrained(self.save_path)

        # Save additional configuration
        config = {
            'labels': self.labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'base_model_name': self.model_name
        }

        with open(os.path.join(self.save_path, 'ner_config.json'), 'w') as f:
            json.dump(config, f)

        print(f"\nModel and configuration saved to {self.save_path}")

        return train_result


def load_synthetic_pii_data(train_data=None, test_data=None):
    """Load and prepare Gretel's synthetic PII finance dataset - English only"""
    if train_data is None or test_data is None:
        dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual")

        # Filter for English texts using the language column
        def is_english(example):
            return example['language'] == 'English'

        # Filter and select samples
        train_data = dataset['train'].filter(is_english)
        test_data = dataset['test'].filter(is_english)

        print(f"Total English training examples found: {len(train_data)}")
        print(f"Total English test examples found: {len(test_data)}")

        # Select requested number of samples
        train_data = train_data.select(range(min(5000, len(train_data))))
        test_data = test_data.select(range(min(500, len(test_data))))

    def prepare_data(examples):
        texts = []
        labels_list = []

        for text, spans in zip(examples['generated_text'], examples['pii_spans']):
            if isinstance(spans, str):
                spans = ast.literal_eval(spans)

            # Filter for only name, company, street_address spans
            # No mapping needed since we're keeping original labels
            target_labels = {'name', 'company', 'street_address'}

            filtered_spans = []
            for span in spans:
                label = span['label']
                if label in target_labels:
                    filtered_spans.append((
                        span['start'],
                        span['end'],
                        label  # Keep original lowercase label
                    ))

            # Only add examples that have at least one relevant entity
            if filtered_spans:
                texts.append(text)
                labels_list.append(filtered_spans)

        return texts, labels_list

    train_texts, train_labels = prepare_data(train_data)
    test_texts, test_labels = prepare_data(test_data)

    print(f"\nFinal statistics after filtering:")
    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples: {len(test_texts)}")

    print("\nEntity distribution in training set:")
    entity_counts = {'name': 0, 'company': 0, 'street_address': 0}
    for labels in train_labels:
        for _, _, label in labels:
            entity_counts[label] += 1
    for entity, count in entity_counts.items():
        print(f"{entity}: {count}")

    # Print a few examples to verify the data
    print("\nSample training examples:")
    for i in range(min(3, len(train_texts))):
        print(f"\nText {i + 1}: {train_texts[i][:500]}...")  # Truncate long texts
        print(f"Labels {i + 1}:")
        for start, end, label in sorted(train_labels[i]):
            print(f"- {label}: '{train_texts[i][start:end]}'")

    return train_texts, train_labels, test_texts, test_labels

def main():
    parser = argparse.ArgumentParser(description='NER Model Training and Prediction')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict',
                       help='Mode to run: train or predict')
    parser.add_argument('--model_path', default='./ner_model',
                       help='Path to save/load the model')
    parser.add_argument('--text', help='Text to predict entities from (for predict mode)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    args = parser.parse_args()

    if args.mode == 'train':
        # Load and prepare data
        print("Loading Gretel's synthetic PII finance dataset...")
        train_texts, train_labels, test_texts, test_labels = load_synthetic_pii_data()

        # Initialize and train the model
        print("Initializing NER model...")
        ner = NERPredictor(model_name="xlm-roberta-base",
                          labels=["O", "name", "company", "street_address"])

        try:
            ner.train(
                train_texts,
                train_labels,
                test_texts,
                test_labels,
                epochs=args.epochs,
                batch_size=16,  # Increased batch size
                learning_rate=1e-5,  # Lower learning rate
                use_cpu=False
            )

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\nGPU out of memory. Switching to CPU training...")
                ner.train(
                    train_texts,
                    train_labels,
                    test_texts,
                    test_labels,
                    epochs=args.epochs,
                    batch_size=8,
                    learning_rate=1e-5,
                    use_cpu=True
                )
            else:
                raise e

    elif args.mode == 'predict':
        if not os.path.exists(args.model_path):
            print(f"Error: Model directory not found at {args.model_path}")
            return

        # Load the model
        print("Loading model...")
        ner = NERPredictor.load_model(args.model_path)

        # Use provided text or default test sentences
        texts_to_predict = []
        if args.text:
            texts_to_predict.append(args.text)
        else:
            texts_to_predict = [
                "John Smith from Apple Inc sent a letter to 123 Main Street, New York.",
                "Microsoft Corporation's CEO Satya Nadella visited the office at 456 Tech Boulevard.",
                "Sarah Johnson works at Deutsche Bank in Frankfurt."
            ]

        print("\nPredicting entities in text:")
        for text in texts_to_predict:
            print(f"\nInput: {text}")
            predictions = ner.predict(text)
            if predictions:
                print("Entities found:")
                for text, label in predictions:
                    print(f"- {text}: {label}")
            else:
                print("No entities found.")


if __name__ == "__main__":
    main()