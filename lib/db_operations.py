import chromadb
from sentence_transformers import SentenceTransformer

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
        
    for subtask in subtasks:
        subtask = subtask.strip()
        if not subtask: continue
        try:
            query_vector = local_embed_model.encode(subtask).tolist()
            results = collection.query(query_embeddings=[query_vector], n_results=1)

            if results['distances'] and len(results['distances'][0]) > 0:
                dist = int(results['distances'][0][0])
                found_task = results['metadatas'][0][0]['task']
                print(f"distance {dist}: ", subtask, "VS", found_task)
                found_doc = results['documents'][0][0]
                if dist <= 25:
                    if found_task not in found_subflows:
                        found_subflows.append(found_task)
                        reference_info += f"\n{found_doc}\n"
        except Exception as e:
            print(f"  [Warning] DB検索中にエラー: {e}")
            continue
    return found_subflows, reference_info
