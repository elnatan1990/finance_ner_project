import requests
import sys
import os

# Add the root project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_news(api_key, query="finance", num_articles=100):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={num_articles}&apiKey={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    return [{"title": art["title"], "content": art["content"]} for art in articles]

if __name__ == "__main__":
    API_KEY = "9edc01e60be8484b9c9d78f30d88f47b"  # Replace with your API key
    news_data = fetch_news(API_KEY)
    print(news_data[:5])
