# Finance NER Project

This project implements a Named Entity Recognition (NER) system specifically designed for financial documents, combined with a RAG (Retrieval Augmented Generation) system for intelligent querying of financial data. The system processes both structured tabular data and unstructured news articles, identifying key entities such as company names, personal names, and addresses.

Project structure:

finance_ner_project/

├── data_fetch/

	│   ├── fetch_tabular.py

	│   ├── fetch_news.py


├── data_storage/

	│   ├── store_tabular.py

	│   ├── store_news.py


├── ner/

	│   ├── ner.py

	│   ├── ner_model

	│   ├── requirements.txt

	│   ├── pip_requirements.txt

	│   ├── README.md


├── rag_system/

	│   ├── rag_agent.py


├── ui/

	│   ├── app.py

	├── requirements.txt

	├── pip_requirements.txt

	└── README.md

## Setup Instructions

### Prerequisites
- Python 3.8+
- MongoDB (running locally on default port 27017)
- NewsAPI key
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finance_ner_project.git
cd finance_ner_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export NEWS_API_KEY="your_news_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

4. Ensure MongoDB is running locally:
```bash
mongod --dbpath /path/to/your/db
```

### Running the System

1. Fetch and store data:
```bash
Fetch data and print only:
python data_fetch/fetch_news.py
python data_fetch/fetch_tabular.py

Store data to mongo:
python data_fetch/store_news.py
python data_fetch/store_tabular.py
```

2. Launch the web interface:
```bash
python ui/app.py
or
python ui/app_with_ner.py
```

## Technical Approach

### Architecture Overview

The system consists of four main components:

1. **Data Collection & Storage**
   - News articles fetched from NewsAPI
   - Synthetic financial data from gretelai/synthetic_pii_finance_multilingual
   - MongoDB for persistent storage

2. **Named Entity Recognition (NER)**
   - Fine-tuned BERT model for financial entity recognition
   - Focuses on three entity types: company names, personal names, and addresses
   - Custom preprocessing pipeline for financial text

3. **RAG System**
   - Uses FAISS for efficient vector similarity search
   - OpenAI embeddings for document representation
   - GPT-3.5-turbo for response generation

4. **User Interface**
   - Gradio-based web interface
   - Real-time entity highlighting
   - Interactive query system

### Trade-offs and Design Decisions

1. **Model Selection**
   - Chose BERT over spaCy for better accuracy despite higher computational cost
   - Used GPT-3.5-turbo instead of GPT-4 for cost-effectiveness

2. **Storage**
   - MongoDB selected for flexibility with unstructured data
   - Local storage vs. cloud for simplicity and data privacy

3. **Vector Search**
   - FAISS chosen over alternatives for better performance with smaller datasets
   - In-memory vs. persistent index trade-off

## Model Details and Hyperparameters

### NER Model

- Base Model: BERT-base-uncased
- Training Data: Custom dataset of 10,000 financial documents
- Key Hyperparameters:
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Epochs: 3
  - Max Sequence Length: 512
  - Dropout: 0.1

### RAG System

- Embedding Model: OpenAI ada-002
- Vector Store: FAISS with cosine similarity
- LLM: GPT-3.5-turbo
- Parameters:
  - Chunk Size: 1000 tokens
  - Chunk Overlap: 200 tokens
  - Top K: 4 documents for retrieval

## Future Work and Improvements

1. **Model Enhancements**
   - Expand entity types to include financial metrics and dates
   - Implement cross-validation for more robust model evaluation
   - Explore domain-specific pre-training

2. **System Scalability**
   - Implement batch processing for large documents
   - Add support for distributed document storage
   - Optimize vector search for larger datasets

3. **User Experience**
   - Add document upload functionality
   - Implement real-time entity visualization
   - Add export functionality for processed documents

4. **Data Quality**
   - Implement data validation pipeline
   - Add support for multiple languages
   - Enhance entity disambiguation

5. **Performance Optimization**
   - Cache frequent queries
   - Implement async processing for better response times
   - Add model quantization for faster inference
