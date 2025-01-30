import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pymongo import MongoClient
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = "your-token-here"

def load_documents_from_mongo_news_only():
    # Connect to MongoDB and fetch articles
    client = MongoClient("mongodb://localhost:27017/")
    db = client.finance_ner
    collection = db.news_articles
    articles = collection.find()

    # Convert articles to LangChain Document format
    documents = [
        Document(
            page_content=article.get("content", ""),
            metadata={"title": article.get("title", "")}
        )
        for article in articles if article.get("content")
    ]
    return documents

def load_documents_from_mongo():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client.finance_ner

    # Load news articles
    news_collection = db.news_articles
    articles = news_collection.find()

    news_documents = [
        Document(
            page_content=article.get("content", ""),
            metadata={
                "title": article.get("title", ""),
                "source": "mongodb",
                "doc_type": "news_article"
            }
        )
        for article in articles if article.get("content")
    ]

    # Load tabular data documents
    tabular_collection = db.tabular_data
    tabular_data = tabular_collection.find()

    tabular_documents = [
        Document(
            page_content=doc.get("generated_text", ""),
            metadata={
                "document_type": doc.get("document_type", ""),
                "document_description": doc.get("document_description", ""),
                "expanded_type": doc.get("expanded_type", ""),
                "expanded_description": doc.get("expanded_description", ""),
                "language": doc.get("language", ""),
                "domain": doc.get("domain", ""),
                "conformance_score": doc.get("conformance_score", 0),
                "quality_score": doc.get("quality_score", 0),
                "toxicity_score": doc.get("toxicity_score", 0),
                "bias_score": doc.get("bias_score", 0),
                "groundedness_score": doc.get("groundedness_score", 0),
                "source": "mongodb",
                "doc_type": "tabular_document"
            }
        )
        for doc in tabular_data if doc.get("generated_text")
    ]

    return news_documents + tabular_documents

def create_rag_system():
    # Load documents from MongoDB NEWS + TABULAR DATA:
    documents = load_documents_from_mongo()

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Replace with your API key
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create RetrievalQA chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)  # Replace with your API key
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa


if __name__ == "__main__":
    rag_system = create_rag_system()
    print("RAG system ready.")
