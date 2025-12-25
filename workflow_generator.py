import os
import argparse
import uuid
import yaml
import json
import re
from datetime import datetime
import copy
import chromadb
import sys
import subprocess
from dotenv import load_dotenv
from litellm import Router
from sentence_transformers import SentenceTransformer

# --- 0. 設定と初期化 ---
load_dotenv("../.env")

# 引数解析
parser = argparse.ArgumentParser(description="Workflow Generator & Refiner")
parser.add_argument("--model", type=str, help="litellm.yaml内の model_name を指定")
parser.add_argument("--user_file", type=str, help="タスクの内容が書かれたテキストファイルのパス")
parser.add_argument("--sys_main_file", type=str, default="prompts/sys_prompt_main.txt", help="System Prompt for main workflow")
parser.add_argument("--sys_sub_file", type=str, default="prompts/sys_prompt_sub.txt", help="System Prompt for sub workflow")
parser.add_argument("--actions_file", type=str, default="prompts/known_actions.json", help="アクション定義ファイルパス")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_depth", type=int, default=2, help="再帰的分解の最大深度")
parser.add_argument("--disable_rag", action='store_true', help="RAG（DB検索）を無効化するフラグ")
parser.add_argument("--disable_sub", action='store_true', help="サブフロー生成を無効化するフラグ")
parser.add_argument("--generate_diagram", action='store_true', help="完了後にdiagram.pyを実行してワークフロー図を生成する")
args = parser.parse_args()

# --- ヘルパー関数群 ---

def load_file_content(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_and_format_actions(filepath, include_types=None):
    if not os.path.exists(filepath):
        print(f"警告: アクションリストファイルが見つかりません: {filepath}")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            actions_data = json.load(f)
        
        if include_types:
            actions_data = [item for item in actions_data if item.get('type') in include_types]

        return "".join([f"* {item['example']}: {item['description']}\n" for item in actions_data])
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
    except:
        return []

def get_valid_action_prefixes(filepath):
    definitions = get_action_definitions(filepath)
    return [item['action'] for item in definitions]

def extract_json(text):
    """MarkdownコードブロックなどからJSON部分を抽出する (強化版)"""
    # <think>ブロックの除去
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # 1. Markdownコードブロックの除去
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if not match:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    
    if match:
        text = match.group(1)
    
    text = text.strip()

    # 2. 通常のパース試行
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. [復旧策A] カンマで区切られたオブジェクト列の場合 ({...}, {...})
    # 全体を [ ] で囲んで再試行
    try:
        print("  [Info] JSON修復試行 A: リスト化 ([...])")
        repaired_text = f"[{text}]"
        return json.loads(repaired_text)
    except json.JSONDecodeError:
        pass

    # 4. [復旧策B] カンマがなく、改行で連続している場合 ({...}\n{...})
    # } { の間にカンマを挿入してリスト化
    try:
        print("  [Info] JSON修復試行 B: 連続オブジェクトの結合")
        # } と { の間に空白や改行がある場合、カンマを挿入
        repaired_text = re.sub(r'}\s*{', '}, {', text)
        repaired_text = f"[{repaired_text}]"
        return json.loads(repaired_text)
    except json.JSONDecodeError:
        pass

    print(f"JSONパースエラー。生テキスト: {text[:200]}...")
    return None

# --- Router & DB 初期化 ---
try:
    with open("../litellm.yaml", "r") as f:
        config = yaml.safe_load(os.path.expandvars(f.read()))
    router = Router(model_list=config.get("model_list", []))
except FileNotFoundError:
    print("エラー: litellm.yaml が見つかりません。")
    exit(1)

if not args.disable_rag:
    print("Embedding model loading...")
    local_embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient(path="./workflow_db")
    collection = client.get_or_create_collection(name="subflows")


# --- B. ワークフロー生成ロジック ---
def generate_workflow(user_prompt_origin, user_prompt_add, sys_prompt, sid=None, next_sid=None):
    print(f"\n>>> 生成プロセス開始")

    # Step 1. タスク分解 + RAG検索
    if not args.disable_rag:
        # タスク分解
        if args.disable_sub:
            known_actions = load_and_format_actions(args.actions_file, include_types=['primitive'])
        else:
            known_actions = load_and_format_actions(args.actions_file, include_types=['primitive', 'complex'])
        
        decomp_prompt = f"""
        <INSTRUCTIONS>
        Please list major and important subtasks of several words to complete the task: "{user_prompt_origin}". 
        </INSTRUCTIONS>
        <CONSTRAINTS>
        * Subtasks MUST be separated by commas.
        * The number of subtasks MUST not exceed 30.
        * DO NOT use numbering or newlines.
        * DO NOT output thinking process.
        * DO NOT output duplicate subtasks.
        * DO NOT use any actions that are not listed below:
        {known_actions}
        </CONSTRAINTS>
        <OUTPUT_FORMAT>
        Here is example of output format. You must adapt the object names and action names to your task.: 
        look around, go to room, focus on object
        </OUTPUT_FORMAT>
        """
        
        try:
            decomp_resp = router.completion(
                model=args.model,
                messages=[{"role": "user", "content": decomp_prompt}],
                temperature=args.temperature
            )
            # カンマで分割
            raw_subtasks = decomp_resp['choices'][0]['message']['content'].split(',')
            
            subtasks = []
            if len(raw_subtasks) > 30:
                subtasks = raw_subtasks[:30]
            else:
                subtasks = raw_subtasks

        except Exception as e:
            print(f"  [Warning] タスク分解中にエラーが発生しました（スキップします）: {e}")
            subtasks = []

        # DB検索
        reference_info = ""
        found_subflows = []
        
        if subtasks:
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
                        if dist < 25:
                            if found_task not in found_subflows:
                                found_subflows.append(found_task)
                                reference_info += f"\n{found_doc}\n"
                except Exception as e:
                    print(f"  [Warning] DB検索中にエラー: {e}")
                    continue

        print("found_subflows:", found_subflows)

    # Step 2. メイン生成
    if not args.disable_rag and reference_info:
        user_prompt = f"""
{user_prompt_origin}

<INSTRUCTIONS>
You can use the structure of the following workflows,
If you use them, adapt the object names and variable names to your environment, and add comment.
{reference_info}
</INSTRUCTIONS>

{user_prompt_add}
    """
    else:
        user_prompt = f"""
{user_prompt_origin}

{user_prompt_add}
    """

    try:
        response = router.completion(
            model=args.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=args.temperature
        )
        print("sys_prompt:", sys_prompt)
        print("user_prompt:", user_prompt)
        
        response_content = response['choices'][0]['message']['content']
        print(response_content)

        data = extract_json(response_content)
        if not data:
            return None

        steps = normalize_workflow_steps(data)

        # 生成されたステップが1つだけで、そのアクションが元のプロンプトと同一の場合
        if (len(steps) == 1 and 
            steps[0].get("step_type") == "action" and
            steps[0].get("action", "").strip() == user_prompt_origin.strip()):
            
            print(f"  [Info] 無限ループの可能性を検出しました。タスク '{user_prompt_origin}' の分解を中止します。")
            return None

        return data

    except Exception as e:
        print(f"  [Error] ワークフロー生成API呼び出し中にエラーが発生しました: {e}")
        return None


# --- C. リファイン（再帰的分解）ロジック ---

def is_primitive_action(action_str, action_definitions):
    """
    アクション文字列を定義リストと照合する。
    最も長くマッチしたアクション定義を探し、そのtypeが'primitive'か'special'であればTrueを返す。
    マッチしない、またはtypeが'complex'ならFalseを返す。
    """
    if not action_str: return True
    
    best_match_def = None
    for action_def in action_definitions:
        prefix = action_def['action']
        if action_str.startswith(prefix):
            if best_match_def is None or len(prefix) > len(best_match_def['action']):
                best_match_def = action_def
    
    if best_match_def:
        action_type = best_match_def.get('type')
        return action_type == 'primitive' or action_type == 'special'

    # 定義リストにないアクション（未知のアクション）の場合
    # 分解対象とするため False を返す
    return False

def normalize_workflow_steps(data):
    """
    Takes a parsed JSON object (list or dict) and returns a list of steps.
    This is to handle varied LLM output.
    """
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # If the dictionary has a "steps" key with a list, return it.
        if "steps" in data and isinstance(data["steps"], list):
            return data["steps"]
        # If the dictionary itself is a single step
        if "step_type" in data and "sid" in data:
            return [data]
        # Find the first value that is a list and return it
        for value in data.values():
            if isinstance(value, list):
                return value
    print(f"  [Warning] Could not extract steps from: {str(data)[:100]}...")
    return []


def sub_workflow_recursive(user_prompt_origin, steps_list, action_definitions, current_depth=0):

    if current_depth >= args.max_depth:
        print("  [Limit] 最大深度に達したため分解を停止します。")
        return steps_list

    new_steps = []
    
    for step in steps_list:
        
        # 1. Actionステップの処理
        if step.get("step_type") == "action":
            action_content = step.get("action", "").strip()
            
            # 定義データ(辞書リスト)を渡す
            if not is_primitive_action(action_content, action_definitions):
                print(f"  [Refine] 分解可能アクション検出: '{action_content}' -> 分解を実行")                
                known_actions = load_and_format_actions(args.actions_file, include_types=['primitive', 'complex'])                
                user_prompt_add_sub =f"""
<TASK>
Your current task is to generate a specific workflow from "{action_content}" action if possible,
 but DO NOT use "{action_content}" itself in the workflow.
DO NOT use any actions that are not listed below:
{known_actions}
</TASK>
                """
                
                sid_to_replace = step.get("sid")
                next_sid_to_replace = step.get("next_sid")

                # 1. Generate raw sub-workflow without stitching
                sub_workflow_data = generate_workflow(
                    action_content, user_prompt_add_sub, sys_prompt_sub
                )
                
                sub_steps = normalize_workflow_steps(sub_workflow_data)

                if sub_steps:
                    # 2. Recursively refine the new sub-workflow
                    refined_sub_steps = sub_workflow_recursive(
                        action_content,
                        sub_steps, 
                        action_definitions,
                        current_depth + 1
                    )
                    
                    # 3. If refinement is successful, perform stitching and unique ID generation
                    if refined_sub_steps:
                        prefix = str(uuid.uuid4())[:4]
                        
                        # Find original entry point to identify it after prefixing
                        all_sids_in_sub = {s.get('sid') for s in refined_sub_steps}
                        all_next_sids_in_sub = {s.get('next_sid') for s in refined_sub_steps if s.get('next_sid')}
                        entry_points = [s for s in refined_sub_steps if s.get('sid') not in all_next_sids_in_sub]
                        original_entry_sid = entry_points[0].get('sid') if entry_points else refined_sub_steps[0].get('sid')

                        # Create a map to prefix all SIDs, ensuring they are unique within the global context
                        sid_map = {s.get('sid'): f"{prefix}_{s.get('sid')}" for s in refined_sub_steps}
                        
                        def get_all_sids_recursively(steps_list):
                            sids = set()
                            for step in steps_list:
                                if "sid" in step:
                                    sids.add(step["sid"])
                                if step.get("step_type") == "for_loop" and "steps" in step and isinstance(step["steps"], list):
                                    sids.update(get_all_sids_recursively(step["steps"]))
                            return sids

                        def apply_sid_map_recursively(steps_list, sid_map):
                            for step in steps_list:
                                # Rename SID
                                if step.get('sid') in sid_map:
                                    step['sid'] = sid_map[step['sid']]
                                # Rename next_sid
                                if step.get('next_sid') in sid_map:
                                    step['next_sid'] = sid_map[step['next_sid']]
                                # Rename branch links
                                if step.get('step_type') == 'branch':
                                    if step.get('next_sid_if_true') in sid_map:
                                        step['next_sid_if_true'] = sid_map[step['next_sid_if_true']]
                                    if step.get('next_sid_if_false') in sid_map:
                                        step['next_sid_if_false'] = sid_map[step['next_sid_if_false']]
                                # Recurse into loops
                                if step.get("step_type") == "for_loop" and "steps" in step and isinstance(step["steps"], list):
                                    apply_sid_map_recursively(step["steps"], sid_map)

                        # Create a map for all SIDs in the entire sub-tree
                        all_sids_in_subtree = get_all_sids_recursively(refined_sub_steps)
                        sid_map = {sid: f"{prefix}_{sid}" for sid in all_sids_in_subtree}
                        
                        # Apply the map recursively to the entire sub-tree
                        apply_sid_map_recursively(refined_sub_steps, sid_map)
                        
                        # Stitch the entry point: replace the new prefixed SID with the original placeholder SID
                        new_entry_sid = sid_map[original_entry_sid]
                        for s in refined_sub_steps:
                            if s.get('sid') == new_entry_sid:
                                s['sid'] = sid_to_replace
                                break
                        
                        # Stitch all exit points: point them to the parent's next step
                        prefixed_sids = set(sid_map.values())
                        for s in refined_sub_steps:
                            if not s.get('next_sid') or s.get('next_sid') not in prefixed_sids:
                                s['next_sid'] = next_sid_to_replace

                        new_steps.extend(refined_sub_steps)
                    else:
                        print("  [Info] サブワークフローの再帰的分解結果が空のため、元のステップを維持します。")
                        new_steps.append(step)
                else:
                    # sub_workflow_data was None (due to loop detection) or empty
                    print("  [Info] サブワークフロー生成失敗(空、形式不正、または無限ループ検出)。元のステップを維持。")
                    new_steps.append(step)
            else:
                new_steps.append(step)

        # 2. For Loop / Branch の処理
        elif step.get("step_type") == "for_loop":
            if "steps" in step and isinstance(step["steps"], list):
                step["steps"] = sub_workflow_recursive(
                    user_prompt_origin,
                    step["steps"], 
                    action_definitions,
                    current_depth
                )
            new_steps.append(step)
            
        else:
            new_steps.append(step)

    # リンク修復
    for i in range(len(new_steps) - 1):
        curr = new_steps[i]
        next_s = new_steps[i+1]
        if curr.get("step_type") == "action":
            if not curr.get("next_sid"):
                curr["next_sid"] = next_s.get("sid", "")

    return new_steps


# --- Main ---
if __name__ == "__main__":
    # 1. アクション定義の取得 (辞書リスト全体を取得)
    action_definitions = get_action_definitions(args.actions_file)

    # 2. prompts
    sys_prompt_main = load_file_content(args.sys_main_file)
    user_prompt_origin = load_file_content(args.user_file)

    # 3. 初期ワークフロー生成
    print("=== Phase 1: 初期ワークフロー生成 ===")

    if args.disable_sub:
        known_actions = load_and_format_actions(args.actions_file, include_types=['primitive', "special"])
        user_prompt_add_main = f"""
    <INSTRUCTIONS>
    You can use only the following actions.
    {known_actions}
    </INSTRUCTIONS>
        """
    else:
        known_actions = load_and_format_actions(args.actions_file, include_types=['primitive', 'complex', "special"])
        user_prompt_add_main = f"""
    <INSTRUCTIONS>
    You can use the following and other actions.
    {known_actions}
    </INSTRUCTIONS>
        """

    initial_workflow_obj = generate_workflow(user_prompt_origin, user_prompt_add_main, sys_prompt_main)
    initial_steps = normalize_workflow_steps(initial_workflow_obj)

    if not initial_steps:
        print("初期生成に失敗しました。終了します。")
        exit(1)
    
    if args.disable_sub:
        final_steps = initial_steps

    else:
        print("\n=== Phase 2: アクション照合とサブフロー生成 ===")
        sys_prompt_sub = load_file_content(args.sys_sub_file)
        final_steps = sub_workflow_recursive(user_prompt_origin, initial_steps, action_definitions)

    # ルートオブジェクトで最終的なワークフローをラップする
    final_workflow = {
        "sid": "root",
        "step_type": "for_loop",
        "iterator": "_",
        "collection": "singleton",
        "steps": final_steps
    }
    
    # --- 結果出力 ---
    print("\n" + "="*30)
    print("FINAL WORKFLOW JSON")
    print("="*30)
    final_output_str = json.dumps(final_workflow, indent=2, ensure_ascii=False)
    print(final_output_str)

    # --- ファイル保存 ---
    user_file_basename = os.path.splitext(os.path.basename(args.user_file))[0] if args.user_file else "default_task"
    output_dir = os.path.join("output", user_file_basename)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    sanitized_model_name = args.model.split(':')[0].split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # ファイル名を構築 (拡張子を.txtに変更)
    initial_filename = os.path.join(output_dir, f"{sanitized_model_name}-initial-{timestamp}.txt")
    final_filename = os.path.join(output_dir, f"{sanitized_model_name}-final-{timestamp}.txt")

    def write_workflow_to_txt(filename, workflow_obj, workflow_title):
        """引数とワークフローを指定されたファイルに書き込む"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                # 引数を書き込む
                f.write("Command line arguments:\n")
                f.write("------------------------\n")
                for arg, value in vars(args).items():
                    f.write(f"{arg}: {value}\n")
                
                f.write("\n\n")

                # ワークフローを書き込む
                f.write(f"{workflow_title}:\n")
                f.write("------------------------\n")
                if workflow_obj:
                    f.write(json.dumps(workflow_obj, indent=2, ensure_ascii=False))
                else:
                    f.write("None")
            print(f"\n✅ {workflow_title}を '{filename}' に保存しました。")
        except Exception as e:
            print(f"\n❌ {workflow_title}の保存中にエラーが発生しました: {e}")

    # 初期ワークフローと引数を保存
    write_workflow_to_txt(initial_filename, initial_workflow_obj, "Initial Workflow")

    # 最終ワークフローと引数を保存
    write_workflow_to_txt(final_filename, final_workflow, "Final Workflow")

    # ダイアグラム生成
    if args.generate_diagram:
        print("\n=== Diagram Generation ===")
        diagram_script = "output/diagram.py"
        
        if os.path.exists(diagram_script):
            try:
                # 同じPythonインタプリタを使用してdiagram.pyを実行
                cmd = [sys.executable, diagram_script, final_filename]
                print(f"Running: {' '.join(cmd)}")
                
                subprocess.run(cmd, check=True)
                print("✅ ダイアグラム生成が完了しました。")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ diagram.py の実行中にエラーが発生しました: {e}")
            except Exception as e:
                print(f"❌ 予期せぬエラーが発生しました: {e}")
        else:
            print(f"❌ '{diagram_script}' が現在のディレクトリに見つかりません。")