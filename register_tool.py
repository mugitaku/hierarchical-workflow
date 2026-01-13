import json
import uuid
import sys
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

# --- Settings ---
DB_PATH = "./workflow_db"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Initialize model (load only once globally to avoid reloading in a loop)
print(f"Loading model... ({EMBEDDING_MODEL_NAME})")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize DB client
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="subflows")

def register_subflow(json_path):
    """Function to register a single JSON file"""
    try:
        if not os.path.exists(json_path):
            print(f"Error: File not found -> {json_path}")
            return False

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task_description = data.get("name")
        steps_obj = data.get("steps")
        # objects may not be required, but will be loaded if present

        if not task_description or not steps_obj:
            print(f"[Skip] Invalid format: {os.path.basename(json_path)} ('name' and 'steps' are required)")
            return False

        # Check for existence (simple check if the same name is already registered)
        existing = collection.get(where={"name": task_description})
        if existing['ids']:
            print(f"[Skip] Already registered: {task_description}")
            return False

        # Vectorize & register
        workflow_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        vector = model.encode(task_description).tolist()
        new_id = str(uuid.uuid4())

        collection.add(
            ids=[new_id],
            embeddings=[vector],
            documents=[workflow_content],
            metadatas=[{"name": task_description}]
        )

        print(f"[Registration successful] {task_description} (from {os.path.basename(json_path)})")
        return True

    except Exception as e:
        print(f"[Error] {os.path.basename(json_path)}: {e}")
        return False

def register_directory(dir_path):
    """Function to register all JSON files in a directory"""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found -> {dir_path}")
        return

    # Search for json files
    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in the specified directory: {dir_path}")
        return

    print(f"\n--- Starting bulk registration for directory: {dir_path} ({len(json_files)} files) ---")
    success_count = 0
    for json_file in json_files:
        if register_subflow(json_file):
            success_count += 1
    
    print("-" * 30)
    print(f"Complete: {success_count} new items registered.")

if __name__ == "__main__":
    target_path = sys.argv[1]
    
    if os.path.isdir(target_path):
        # If it's a directory
        register_directory(target_path)
    else:
        # If it's a file
        register_subflow(target_path)
