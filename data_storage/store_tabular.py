from pymongo import MongoClient
from data_fetch.fetch_tabular import fetch_tabular_data

def store_tabular_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client.finance_ner
    tabular_data = fetch_tabular_data()

    # Initialize an empty list to store the first 100 items
    first_items = []

    # Use a for loop to go over the data and extract the first 100 items
    for i, item in enumerate(tabular_data):
        if i < 100:
            first_items.append(item)
        else:
            break

    # Insert the first items into MongoDB
    if first_items:
        db.tabular_data.insert_many(first_items)
        print("First 100 items stored in MongoDB.")
    else:
        print("No data to store.")

if __name__ == "__main__":
    store_tabular_data()
