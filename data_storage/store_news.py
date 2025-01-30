from pymongo import MongoClient
from data_fetch.fetch_news import fetch_news

def store_news(api_key):
    client = MongoClient("mongodb://localhost:27017/")
    db = client.finance_ner
    news_data = fetch_news(api_key)
    db.news_articles.insert_many(news_data)

if __name__ == "__main__":
    API_KEY = "9edc01e60be8484b9c9d78f30d88f47b"  # Replace with your API key
    store_news(API_KEY)
    print("News articles stored in MongoDB.")
