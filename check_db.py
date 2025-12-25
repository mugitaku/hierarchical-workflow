import chromadb
import json

# データベースのパス（Generatorの設定と合わせる）
DB_PATH = "./workflow_db"

def check_database():
    try:
        # 1. DBへの接続
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name="subflows")
        
        # 2. データ件数の取得
        count = collection.count()
        print(f"=== データベースステータス ===")
        print(f"保存場所: {DB_PATH}")
        print(f"登録件数: {count} 件")
        print("=" * 30)

        if count == 0:
            print("データは空です。")
            return

        # 3. データの取得 (全件取得)
        # ※件数が多い場合は limit=10 などを指定してください
        data = collection.get()

        print("\n=== 登録データ一覧 ===")
        for i in range(len(data['ids'])):
            print(f"ID: {data['ids'][i]}")
            
            # Metadata (タスク名など)
            metadata = data['metadatas'][i]
            print(f"Task Name: {metadata.get('task', 'N/A')}")
            
            # Document (中身のJSONテキスト)
            # 長すぎる場合は先頭だけ表示して省略
            doc_content = data['documents'][i]
            if len(doc_content) > 200:
                print(f"Content: {doc_content[:200]} ... (省略)")
            else:
                print(f"Content: {doc_content}")
            
            print("-" * 30)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("ヒント: データベースフォルダが存在しないか、パスが間違っている可能性があります。")

if __name__ == "__main__":
    check_database()