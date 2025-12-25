import json
import uuid
import sys
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

# --- 設定 ---
DB_PATH = "./workflow_db"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# モデルの初期化 (ループ内で毎回ロードしないようにグローバルで一度だけ行う)
print(f"モデル読み込み中... ({EMBEDDING_MODEL_NAME})")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# DBクライアントの初期化
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="subflows")

def register_subflow(json_path):
    """単一のJSONファイルを登録する関数"""
    try:
        if not os.path.exists(json_path):
            print(f"エラー: ファイルが見つかりません -> {json_path}")
            return False

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task_description = data.get("task")
        steps_obj = data.get("steps")
        # objects は必須ではないかもしれませんが、存在すれば読み込まれます

        if not task_description or not steps_obj:
            print(f"[スキップ] フォーマット不正: {os.path.basename(json_path)} ('task' と 'steps' が必要)")
            return False

        # 既存チェック (同じタスク名がすでに登録されていないか簡易チェック)
        existing = collection.get(where={"task": task_description})
        if existing['ids']:
            print(f"[スキップ] 登録済み: {task_description}")
            return False

        # ベクトル化 & 登録
        workflow_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        vector = model.encode(task_description).tolist()
        new_id = str(uuid.uuid4())

        collection.add(
            ids=[new_id],
            embeddings=[vector],
            documents=[workflow_content],
            metadatas=[{"task": task_description}]
        )

        print(f"[登録成功] {task_description} (from {os.path.basename(json_path)})")
        return True

    except Exception as e:
        print(f"[エラー] {os.path.basename(json_path)}: {e}")
        return False

def register_directory(dir_path):
    """ディレクトリ内の全JSONを登録する関数"""
    if not os.path.isdir(dir_path):
        print(f"エラー: ディレクトリが見つかりません -> {dir_path}")
        return

    # jsonファイルを検索
    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    
    if not json_files:
        print(f"指定されたディレクトリにJSONファイルがありません: {dir_path}")
        return

    print(f"\n--- ディレクトリ一括登録開始: {dir_path} ({len(json_files)}ファイル) ---")
    success_count = 0
    for json_file in json_files:
        if register_subflow(json_file):
            success_count += 1
    
    print("-" * 30)
    print(f"完了: {success_count} 件を新規登録しました。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方:")
        print("  単一ファイル: python register_tool.py subflows/my_task.json")
        print("  フォルダ一括: python register_tool.py subflows/")
    else:
        target_path = sys.argv[1]
        
        if os.path.isdir(target_path):
            # ディレクトリの場合
            register_directory(target_path)
        else:
            # ファイルの場合
            register_subflow(target_path)