import chromadb
import json

# Database path (must match Generator's settings)
DB_PATH = "./workflow_db"

def check_database():
    try:
        # 1. Connect to DB
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name="subflows")
        
        # 2. Get data count
        count = collection.count()
        print(f"=== Database Status ===")
        print(f"Location: {DB_PATH}")
        print(f"Number of entries: {count}")
        print("=" * 30)

        if count == 0:
            print("The database is empty.")
            return

        # 3. Get data (get all)
        # * If there are many items, specify something like limit=10
        data = collection.get()

        print("\n=== Registered Data List ===")
        for i in range(len(data['ids'])):
            print(f"ID: {data['ids'][i]}")
            
            # Metadata (task name, etc.)
            metadata = data['metadatas'][i]
            print(f"Task Name: {metadata.get('name', 'N/A')}")
            
            # Document (JSON text inside)
            # If too long, display only the beginning and omit the rest
            doc_content = data['documents'][i]
            if len(doc_content) > 200:
                print(f"Content: {doc_content[:200]} ... (omitted)")
            else:
                print(f"Content: {doc_content}")
            
            print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Hint: The database folder may not exist or the path may be incorrect.")

if __name__ == "__main__":
    check_database()
