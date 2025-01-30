# Named Entity Recognition (NER) Model for PII Detection

This repository contains a Named Entity Recognition (NER) system 
built using XLM-RoBERTa for detecting Personal Identifiable Information (PII) in text. 
The model is specifically trained to identify names, company names, and street addresses.

## Setup

### Prerequisites

```bash
pip install torch
pip install transformers
pip install datasets
pip install seqeval
pip install numpy
```

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ner-pii-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new model:

```bash
python main.py --mode train --model_path ./ner_model --epochs 5
```

### Prediction

To run predictions on text:

```bash
python main.py --mode predict --model_path ./ner_model --text "Your text here"
```

## Technical Approach

### Model Architecture

- Base Model: XLM-RoBERTa
- Task: Token Classification (NER)
- Output Labels: name, company, street_address

The system uses XLM-RoBERTa as the backbone model, fine-tuned for token classification. This choice was made for several reasons:

1. Multilingual Capability: Although currently trained on English data, the model can be extended to other languages.
2. Strong Contextual Understanding: RoBERTa's architecture is particularly effective at capturing contextual information.
3. State-of-the-art Performance: XLM-RoBERTa has shown superior performance on various NLP tasks.

### Data Processing

- Dataset: Gretel's synthetic PII finance dataset
- Training Size: Up to 5000 examples
- Validation Size: Up to 500 examples
- Entity Types: name, company, street_address

The data processing pipeline includes:
1. Filtering for English-language examples
2. Converting span annotations to token-level labels
3. Handling special tokens and offset mapping
4. Dynamic batch creation with padding

### Training Configuration

Key hyperparameters:

```python
{
    "learning_rate": 1e-5,
    "batch_size": 16,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2
}
```

These hyperparameters were chosen based on:
- Learning Rate: Lower learning rate (1e-5) for stable fine-tuning
- Batch Size: Optimized for memory efficiency while maintaining training stability
- Gradient Accumulation: Helps simulate larger batch sizes
- Warmup & Weight Decay: Prevents early training instability and overfitting

## Performance Metrics

The model is evaluated using:
- Macro F1 Score: Overall model performance
- Per-entity Precision, Recall, and F1 scores
- Entity-specific metrics for detailed performance analysis

### Sample Metrics Output:

```
Evaluation Metrics:
Macro F1: 0.8934

name          - F1: 0.9123, Precision: 0.9245, Recall: 0.9004
company       - F1: 0.8876, Precision: 0.8932, Recall: 0.8821
street_address- F1: 0.8803, Precision: 0.8756, Recall: 0.8851
```

## Future Improvements

1. Model Enhancement:
   - Experiment with different transformer architectures (DeBERTa, BERT-Large)
   - Implement ensemble methods for improved robustness
   - Add support for additional PII entity types

2. Training Optimization:
   - Implement cross-validation for more robust evaluation
   - Experiment with dynamic learning rates and scheduling
   - Add data augmentation techniques

3. Performance Improvements:
   - Optimize inference speed through model quantization
   - Implement batch processing for large-scale text analysis
   - Add caching mechanisms for frequent predictions

4. Additional Features:
   - Add confidence scores for predictions
   - Implement active learning for continuous model improvement
   - Add support for more languages using the multilingual capability

5. Robustness:
   - Add handling for edge cases and unusual text formats
   - Improve performance on noisy text data
   - Add better error handling and recovery mechanisms

## Limitations

1. Data Constraints:
   - Currently trained primarily on synthetic data
   - Limited to three entity types
   - English-language focus

2. Model Constraints:
   - Maximum sequence length of 512 tokens
   - Resource-intensive during training
   - Potential for false positives in ambiguous cases

## Citation

If you use this code, please cite:

```bibtex
@misc{ner-pii-detection,
  author = {[Your Name]},
  title = {Named Entity Recognition for PII Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[repository-url]}}
}
```