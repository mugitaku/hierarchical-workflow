import chromadb
from sentence_transformers import SentenceTransformer
import json
import uuid

def initialize_db(disable_db):
    if disable_db:
        return None, None
    print("Embedding model loading...")
    local_embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient(path="./workflow_db")
    collection = client.get_or_create_collection(name="subflows")
    return local_embed_model, collection

def search_subflows(collection, local_embed_model, subtasks, disable_db):
    found_subflows = []
    reference_info = ""
    if not subtasks or disable_db:
        return found_subflows, reference_info
        
    for original_subtask in subtasks[:]:
        subtask = original_subtask.strip()
        if not subtask: continue
        try:
            query_vector = local_embed_model.encode(subtask).tolist()
            results = collection.query(query_embeddings=[query_vector], n_results=1)

            if results['distances'] and len(results['distances'][0]) > 0:
                dist = results['distances'][0][0]
                found_task = results['metadatas'][0][0]['name']
                print(f"distance {int(dist)}: ", subtask, "VS", found_task)
                found_doc = results['documents'][0][0]
                if dist <= 25:
                    # remove subtask from subtasks list
                    subtasks.remove(original_subtask)
                    
                    # add subflow candidates to list
                    if found_task not in found_subflows:
                        found_subflows.append(found_task)
                        reference_info += f"\n{found_doc}\n"
        except Exception as e:
            print(f"  [Warning] Error during DB search: {e}")
            continue
    return found_subflows, reference_info

def get_all_subflows(collection):
    """Fetches all subflow documents from the database."""
    if not collection:
        return []
    try:
        results = collection.get(include=["documents"])
        return results.get('documents', [])
    except Exception as e:
        print(f"  [Warning] Error during fetching all subflows: {e}")
        return []

def register_subflow_in_db(collection, model, subflow_data):
    """Registers a single subflow dictionary in the database."""
    if not collection or not model:
        return False

    task_description = subflow_data.get("name")
    steps_obj = subflow_data.get("steps")

    if not task_description or not steps_obj:
        print(f"[Skip] Invalid format: 'name' and 'steps' are required")
        return False

    # Vectorize & register
    workflow_content = json.dumps(subflow_data, indent=2, ensure_ascii=False)
    
    vector = model.encode(task_description).tolist()
    new_id = str(uuid.uuid4())

    try:
        collection.add(
            ids=[new_id],
            embeddings=[vector],
            documents=[workflow_content],
            metadatas=[{"name": task_description}]
        )
        print(f"[Registration successful] {task_description}")
        return True
    except Exception as e:
        print(f"[Error] on registration: {e}")
        return False
