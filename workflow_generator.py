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
import time
import random
import html
from dotenv import load_dotenv
from litellm import Router
from sentence_transformers import SentenceTransformer

# --- 0. 設定と初期化 ---
load_dotenv("../.env")

# 引数解析
parser = argparse.ArgumentParser(description="Workflow Generator & Refiner")
parser.add_argument("--model", type=str, help="litellm.yaml内の model_name を指定")
parser.add_argument("--user_file", type=str, help="タスクの内容が書かれたテキストファイルのパス")
parser.add_argument("--sys_main_file", type=str, default="prompts/sys/main.txt", help="System Prompt for main workflow")
parser.add_argument("--sys_sub_file", type=str, default="prompts/sys/sub.txt", help="System Prompt for sub workflow")
parser.add_argument("--format_file", type=str, default="prompts/sys/format.txt", help="System Prompt for format")
parser.add_argument("--example_file", type=str, default="prompts/sys/example.txt", help="System Prompt for few shot example")
parser.add_argument("--actions_file", type=str, default="prompts/known_actions.json", help="アクション定義ファイルパス")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_depth", type=int, default=2, help="再帰的分解の最大深度")
parser.add_argument("--disable_example", action='store_true', help="few shot exampleを無効化するフラグ")
parser.add_argument("--disable_db", action='store_true', help="DB検索を無効化するフラグ")
parser.add_argument("--disable_subflow", action='store_true', help="サブフロー生成を無効化するフラグ")
parser.add_argument("--generate_diagram", action='store_true', help="完了後にdiagram.pyを実行してワークフロー図を生成する")
args = parser.parse_args()

# --- ヘルパー関数群 ---

def load_file_content(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_and_format_actions(filepath, types=None):
    if not os.path.exists(filepath):
        print(f"警告: アクションリストファイルが見つかりません: {filepath}")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            actions_data = json.load(f)
        
        if types:
            actions_data = [item for item in actions_data if item.get('type') in types]

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
    """MarkdownコードブロックなどからJSON部分を抽出する (強化版, v4)"""
    # Unescape HTML entities
    text = html.unescape(text)

    # <think>ブロックの除去
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # <steps>タグがあれば抽出
    steps_match = re.search(r'<steps>\s*(.*?)\s*</steps>', text, re.DOTALL)
    if steps_match:
        text = steps_match.group(1).strip()

    # Markdownコードブロックの抽出
    code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if not code_block_match:
        code_block_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    
    if code_block_match:
        text = code_block_match.group(1)
    
    # コメント削除 (行末のカンマを巻き込まないように注意)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # 行コメント削除: // の後にカンマがある場合を救済するため、単純削除ではなく改行のみ残す等の工夫も可能だが
    # ここではシンプルに削除しつつ、パース前処理でカンマ補完を試みる
    text = re.sub(r'//.*', '', text)

    clean_text = text.strip()

    # JSONの開始・終了位置を特定
    start_pos = -1
    first_brace = clean_text.find('{')
    first_bracket = clean_text.find('[')
    if first_brace == -1: start_pos = first_bracket
    elif first_bracket == -1: start_pos = first_brace
    else: start_pos = min(first_brace, first_bracket)

    if start_pos == -1:
        print("JSONの開始文字 ('{' または '[') が見つかりません。")
        return None

    end_pos = -1
    last_brace = clean_text.rfind('}')
    last_bracket = clean_text.rfind(']')
    end_pos = max(last_brace, last_bracket)

    if end_pos == -1:
        print("JSONの終了文字 ('}' または ']') が見つかりません（出力が途切れている可能性があります）。")
        return None

    json_candidate_str = clean_text[start_pos:end_pos+1]

    # 一般的なエラー（プロパティ間のカンマ漏れ）の修正試行
    json_str_fixed = re.sub(r'"\s*\n\s*"', '",\n"', json_candidate_str)

    try:
        return json.loads(json_str_fixed)
    except json.JSONDecodeError:
        try:
            return json.loads(json_candidate_str)
        except json.JSONDecodeError as e1:
            # Attempt 3: 二重括弧 {{...}} または [[...]] の場合のみ剥がすように変更
            # 単一の {...} を剥がすと Extra data エラーになるため防止する
            if (json_candidate_str.startswith('{{') and json_candidate_str.endswith('}}')) or \
               (json_candidate_str.startswith('[[') and json_candidate_str.endswith(']]')):
                inner_str = json_candidate_str[1:-1]
                try:
                    print("  [Info] 二重括弧を検出しました。外側を剥がして再試行します。")
                    return json.loads(inner_str)
                except json.JSONDecodeError:
                    pass
            
            print(f"JSONパースエラー。リカバリー不能。エラー: {e1}")
            # デバッグ用に末尾を表示（途切れ確認用）
            print(f"文字列末尾(last 100 chars): ...{json_candidate_str[-100:]}")
            return None

# --- Router & DB 初期化 ---
try:
    with open("../litellm.yaml", "r") as f:
        config = yaml.safe_load(os.path.expandvars(f.read()))
    router = Router(model_list=config.get("model_list", []))
except FileNotFoundError:
    print("エラー: litellm.yaml が見つかりません。")
    exit(1)

if not args.disable_db:
    print("Embedding model loading...")
    local_embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient(path="./workflow_db")
    collection = client.get_or_create_collection(name="subflows")


# --- リトライ機能付きAPI呼び出し関数 ---
def completion_with_backoff(**kwargs):
    """
    RateLimitError発生時に指数関数的バックオフでリトライを行うラッパー関数
    """
    max_retries = 8  # リトライ回数を増やす
    base_delay = 5   # 初回待機時間（秒）
    
    if "timeout" not in kwargs:
        kwargs["timeout"] = 120
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 8192

    for attempt in range(max_retries + 1):
        try:
            return router.completion(**kwargs)
        except Exception as e:
            error_str = str(e)
            if ("RateLimitError" in error_str or 
                "429" in error_str or 
                "ServiceUnavailableError" in error_str or 
                "Timeout" in error_str): # タイムアウト関連のエラーをキャッチ
                
                if attempt < max_retries:
                    # 指数関数的バックオフ + ランダムなゆらぎ (Jitter)
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"  [Retry] エラー発生: {e}")
                    print(f"  -> {delay:.2f}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            # その他のエラー、またはリトライ上限到達時は例外を投げる
            raise e


# --- ワークフロー生成ロジック ---
def generate_workflow(sys_prompt, user_prompt_origin, user_prompt_add):
    print(f"\n>>> 生成プロセス開始")
    # Step 1. Task Decomposition
    reference_info = ""
    # primitive1 is routine action, so it is not used here.
    if args.disable_db:
        known_actions = load_and_format_actions(args.actions_file, types=['primitive2'])
    else:
        known_actions = load_and_format_actions(args.actions_file, types=['primitive2', 'complex'])
    
    decomp_prompt = f"""
    <INSTRUCTIONS>
    List major subtasks to complete the following task: "{user_prompt_origin}". 
    Abstract all the object names of the subtasks (e.g. "cup" is abstracted to "container", "stove" is abstracted to "heater", etc)
    </INSTRUCTIONS>
    <CONSTRAINTS>
    * Subtasks MUST be separated by commas.
    * Subtasks MUST be 10 words or less.
    * The number of subtasks MUST be between 20 and 30.
    * DO NOT use numbering or newlines.
    * DO NOT output thinking process.
    * DO NOT output duplicate subtasks.
    * DO NOT use any actions that are not listed below.:
    {known_actions}
    </CONSTRAINTS>
    <OUTPUT_FORMAT>
    subtask1, subtask2, ...
    </OUTPUT_FORMAT>
    """
    
    try:
        decomp_resp = completion_with_backoff(
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

    # Step 2. DB検索
    if not args.disable_db:
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
                        if dist <= 25:
                            if found_task not in found_subflows:
                                found_subflows.append(found_task)
                                reference_info += f"\n{found_doc}\n"
                except Exception as e:
                    print(f"  [Warning] DB検索中にエラー: {e}")
                    continue


    # Step 3. メイン生成
    if reference_info:
        user_prompt = f"""
{user_prompt_origin}
<INSTRUCTIONS>
Refer to the following subtasks and workflows, adapting the object names and variable names to your task.
subtasks: {subtasks}
workflows:
{reference_info}
</INSTRUCTIONS>
{user_prompt_add}
    """
    else:
        user_prompt = f"""
{user_prompt_origin}
<INSTRUCTIONS>
Refer to the following subtasks, adapting the object names and variable names to your task.
{subtasks}
</INSTRUCTIONS>
{user_prompt_add}
    """

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=args.temperature
        )
        print("■sys_prompt:", sys_prompt)
        print("■user_prompt:", user_prompt)
        
        response_content = response['choices'][0]['message']['content']
        print("■response_content:", response_content)

        data = extract_json(response_content)
        if not data:
            return None

        return data

    except Exception as e:
        print(f"  [Error] ワークフロー生成API呼び出し中にエラーが発生しました: {e}")
        return None


# --- リファイン（再帰的分解）ロジック ---
def is_decomposable(action_str, action_definitions):
    """
    アクション文字列を定義リストと照合し、分解可能かどうかを判断する。
    最も長くマッチしたアクション定義を探し、そのtypeが'complex'か、または定義が見つからなければTrueを返す。
    typeが'primitive1', 'primitive2', 'special'であればFalseを返す。
    """
    if not action_str: return False # Empty action is not decomposable
    
    best_match_def = None
    for action_def in action_definitions:
        prefix = action_def['action']
        if action_str.startswith(prefix):
            if best_match_def is None or len(prefix) > len(best_match_def['action']):
                best_match_def = action_def
    
    if best_match_def:
        action_type = best_match_def.get('type')
        # 'complex' is decomposable. 'primitive1', 'primitive2', and 'special' are not.
        return action_type == 'complex'

    # No definition found, so we assume it's a complex action that needs decomposition.
    return True

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


def refine_final_workflow(steps, user_prompt_origin, format_content):
    print("\n=== Phase 3: 最終ワークフローの最適化 ===")
    
    workflow_json_str = json.dumps(steps, indent=2, ensure_ascii=False)

    refine_prompt = f"""
<INSTRUCTIONS>
Refine the workflow showed in WORKFLOW_TO_REFINE section.
The workflow is to complete the task showed in OVERALL_TASK section.

Follow these instructions to refine the workflow:
* Insert missing steps for the workflow to be successfully executed
* Replace ambiguous action steps with more specific ones
* Adapt the object names and variable names in the steps to your environment written in CONTEXT section
* Use "go to <room>" action step properly
* Conform the workflow to output format
    * All steps have unique `sid`
    * All `sid` and `next_sid` links are consistent
    * If a action step contains multimle actions, split it into multiple steps
    * Allowed step_type = ['action', 'branch', "for_loop", "break"]
</INSTRUCTIONS>

{format_content}

{user_prompt_origin}

<WORKFLOW_TO_REFINE>
{workflow_json_str}
</WORKFLOW_TO_REFINE>

<CONSTRAINTS>
DO NOT use any actions that are not listed below.:
{load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', "special"])}
</CONSTRAINTS>
"""

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[
                {"role": "user", "content": refine_prompt}
            ],
            temperature=args.temperature
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response:", response_content)

        refined_data = extract_json(response_content)
        if not refined_data:
            print("  [Warning] 最終リファインに失敗しました。元のワークフローを維持します。")
            return steps # Return original steps on failure

        refined_steps = normalize_workflow_steps(refined_data)
        return refined_steps

    except Exception as e:
        print(f"  [Error] 最終リファイン中にエラーが発生しました: {e}")
        return steps # Return original steps on error


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
            if is_decomposable(action_content, action_definitions):
                if current_depth >= args.max_depth - 1:
                    action_constraints = "You can use only the following actions:" + "\n" +load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2'])
                elif args.disable_db:
                    action_constraints = "You can use the following and other actions:" + "\n" +load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2'])
                else:
                    action_constraints = "You can use only the following actions:" + "\n" +load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', 'complex'])
                
                user_prompt_add_sub =f"""
<SPECIFIC_TASK>
Generate workflow from "{action_content}" action if possible,
 but DO NOT use "{action_content}" itself in the workflow.
The object names and action names in the workflow must be adapted to your task.
{action_constraints}
</SPECIFIC_TASK>
                """
                
                sid_to_replace = step.get("sid")
                next_sid_to_replace = step.get("next_sid")

                # 1. Generate raw sub-workflow without stitching
                sub_workflow_data = generate_workflow(sys_prompt_sub, user_prompt_origin, user_prompt_add_sub)                
                
                sub_steps = normalize_workflow_steps(sub_workflow_data)

                if sub_steps:
                    # 2. Recursively generate the new sub-workflow
                    generated_sub_steps = sub_workflow_recursive(
                        user_prompt_origin,
                        sub_steps, 
                        action_definitions,
                        current_depth + 1
                    )
                    
                    # 3. If generation is successful, perform stitching and unique ID generation
                    if generated_sub_steps:
                        prefix = str(uuid.uuid4())[:4]
                        
# Find original entry point to identify it after prefixing
                        all_sids_in_sub = {s.get('sid') for s in generated_sub_steps}
                        
                        all_next_sids_in_sub = set()
                        for s in generated_sub_steps:
                            if s.get('next_sid'):
                                all_next_sids_in_sub.add(s.get('next_sid'))
                            if s.get('step_type') == 'branch':
                                if s.get('next_sid_if_true'):
                                    all_next_sids_in_sub.add(s.get('next_sid_if_true'))
                                if s.get('next_sid_if_false'):
                                    all_next_sids_in_sub.add(s.get('next_sid_if_false'))
                        
                        entry_points = [s for s in generated_sub_steps if s.get('sid') not in all_next_sids_in_sub]
                        original_entry_sid = entry_points[0].get('sid') if entry_points else generated_sub_steps[0].get('sid')

                        # Create a map to prefix all SIDs, ensuring they are unique within the global context
                        sid_map = {s.get('sid'): f"{prefix}_{s.get('sid')}" for s in generated_sub_steps}
                        
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
                        all_sids_in_subtree = get_all_sids_recursively(generated_sub_steps)
                        sid_map = {sid: f"{prefix}_{sid}" for sid in all_sids_in_subtree}
                        
                        # Apply the map recursively to the entire sub-tree
                        apply_sid_map_recursively(generated_sub_steps, sid_map)
                        
                        # Stitch the entry point: replace the new prefixed SID with the original placeholder SID
                        new_entry_sid = sid_map[original_entry_sid]
                        for s in generated_sub_steps:
                            if s.get('sid') == new_entry_sid:
                                s['sid'] = sid_to_replace
                                break
                        
                        # Stitch all exit points: point them to the parent's next step
                        prefixed_sids = set(sid_map.values())
                        for s in generated_sub_steps:
                            if not s.get('next_sid') or s.get('next_sid') not in prefixed_sids:
                                s['next_sid'] = next_sid_to_replace

                        new_steps.extend(generated_sub_steps)
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

    return new_steps


# --- Main ---
if __name__ == "__main__":
    # 1. アクション定義の取得 (辞書リスト全体を取得)
    action_definitions = get_action_definitions(args.actions_file)

    # 2. prompts
    format_content = load_file_content(args.format_file)
    example_content = load_file_content(args.example_file)
    if args.disable_example:
        sys_prompt_main = load_file_content(args.sys_main_file) + "\n" + format_content
    else:
        sys_prompt_main = load_file_content(args.sys_main_file) + "\n" + format_content + "\n" + example_content
    user_prompt_origin = load_file_content(args.user_file)

    # 3. 初期ワークフロー生成
    print("=== Phase 1: 初期ワークフロー生成 ===")

    if args.disable_subflow:
        user_prompt_add_main = f"""
    <INSTRUCTIONS>
    You can use only the following actions:
    {load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', "special"])}
    </INSTRUCTIONS>
        """
    elif args.disable_db:
        user_prompt_add_main = f"""
    <INSTRUCTIONS>
    You can use the following and other actions:
    {load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', "special"])}
    </INSTRUCTIONS>
        """
    else:
        user_prompt_add_main = f"""
    <INSTRUCTIONS>
    You can use the following and other actions:
    {load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', 'complex', "special"])}
    </INSTRUCTIONS>
        """

    initial_workflow_obj = generate_workflow(sys_prompt_main, user_prompt_origin, user_prompt_add_main)
    initial_steps = normalize_workflow_steps(initial_workflow_obj)

    if not initial_steps:
        print("初期生成に失敗しました。終了します。")
        exit(1)

    initial_workflow_wrapped = {
        "sid": "root",
        "step_type": "for_loop",
        "iterator": "_",
        "collection": "singleton",
        "steps": initial_steps
    }
    
    if args.disable_subflow:
        final_steps = initial_steps
        pre_refined_final_steps = initial_steps
    else:
        print("\n=== Phase 2: アクション照合とサブフロー生成 ===")
        if args.disable_example:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content
        else:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content + "\n" + example_content
        
        pre_refined_final_steps = sub_workflow_recursive(user_prompt_origin, copy.deepcopy(initial_steps), action_definitions)
        final_steps = pre_refined_final_steps

    pre_refined_workflow = {
        "sid": "root",
        "step_type": "for_loop",
        "iterator": "_",
        "collection": "singleton",
        "steps": pre_refined_final_steps
    }

    final_steps = refine_final_workflow(final_steps, user_prompt_origin, format_content)

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
    initial_filename = os.path.join(output_dir, f"{sanitized_model_name}-1-initial-{timestamp}.txt")
    pre_refine_filename = os.path.join(output_dir, f"{sanitized_model_name}-2-pre-refine-{timestamp}.txt")
    final_filename = os.path.join(output_dir, f"{sanitized_model_name}-3-final-{timestamp}.txt")

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
    write_workflow_to_txt(initial_filename, initial_workflow_wrapped, "Initial Workflow")

    # リファイン前ワークフローと引数を保存
    write_workflow_to_txt(pre_refine_filename, pre_refined_workflow, "Pre-Refined Final Workflow")

    # 最終ワークフローと引数を保存
    write_workflow_to_txt(final_filename, final_workflow, "Final Workflow")

    # ダイアグラム生成
    if args.generate_diagram:
        print("\n=== Diagram Generation ===")
        diagram_script = "diagram.py"
        
        if os.path.exists(diagram_script):
            filenames_to_diagram = {
                "Initial": initial_filename,
                "Pre-Refined": pre_refine_filename,
                "Final": final_filename
            }
            
            for name, filename in filenames_to_diagram.items():
                try:
                    # 同じPythonインタプリタを使用してdiagram.pyを実行
                    cmd = [sys.executable, diagram_script, filename]
                    print(f"\n--- Generating diagram for {name} workflow ---")
                    print(f"Running: {' '.join(cmd)}")
                    
                    subprocess.run(cmd, check=True)
                    print(f"✅ {name} のダイアグラム生成が完了しました。")
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ {name} の diagram.py の実行中にエラーが発生しました: {e}")
                except Exception as e:
                    print(f"❌ {name} のダイアグラム生成中に予期せぬエラーが発生しました: {e}")
        else:
            print(f"❌ '{diagram_script}' が現在のディレクトリに見つかりません。")
