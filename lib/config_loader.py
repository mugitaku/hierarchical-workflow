import argparse
import os
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Workflow Generator & Refiner")
    parser.add_argument("--model", type=str, help="litellm.yaml内の model_name を指定")
    parser.add_argument("--user_env_file", type=str, help="エージェントの環境が書かれたテキストファイル")
    parser.add_argument("--user_task_file", type=str, help="エージェントのタスクが書かれたテキストファイル")
    parser.add_argument("--sys_main_file", type=str, default="prompts/sys/main.txt", help="System Prompt for main workflow")
    parser.add_argument("--sys_sub_file", type=str, default="prompts/sys/sub.txt", help="System Prompt for sub workflow")
    parser.add_argument("--format_file", type=str, default="prompts/sys/format.txt", help="System Prompt for format")
    parser.add_argument("--example_file", type=str, default="prompts/sys/example.txt", help="System Prompt for few shot example")
    parser.add_argument("--actions_file", type=str, default="prompts/known_actions-simple.json", help="アクション定義ファイルパス")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_depth", type=int, default=1, help="再帰的分解の最大深度")
    parser.add_argument("--disable_example", action='store_true', help="few shot exampleを無効化するフラグ")
    parser.add_argument("--disable_db", action='store_true', help="DB検索を無効化するフラグ")
    parser.add_argument("--disable_subflow", action='store_true', help="サブフロー生成を無効化するフラグ")
    parser.add_argument("--generate_diagram", action='store_true', help="完了後にdiagram.pyを実行してワークフロー図を生成する")
    return parser.parse_args()

def load_file_content(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_and_format_actions(filepath, action_limited=False, action_types=None):
    if not os.path.exists(filepath):
        print(f"警告: アクションリストファイルが見つかりません: {filepath}")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            actions_data = json.load(f)
        
        if types:
            actions_data = [item for item in actions_data if item.get('type') in types]

        header = "You can use only the following actions:\n" if limited else "You can use the following and other actions:\n"
        actions_str = "".join([f"* {item['example']}: {item['description']}\n" for item in actions_data])
        return header + actions_str
    except Exception as e:
        print(f"アクションリスト読込エラー: {e}")
        return ""

def get_action_definitions(filepath):
    """actions_list.json全体を辞書リストとして取得する"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"アクション定義ファイルのJSONデコードエラー: {e}")
        return []
