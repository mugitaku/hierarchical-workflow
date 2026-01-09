import re
import json
import html

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

def get_all_sids(steps):
    """Recursively traverses the workflow to collect all sids."""
    sids = set()
    if not isinstance(steps, list):
        return sids
    for step in steps:
        if 'sid' in step:
            sids.add(step['sid'])
        if step.get('step_type') == 'for_loop' and 'steps' in step:
            sids.update(get_all_sids(step['steps']))
    return sids


def find_duplicate_sids(steps):
    """
    Finds duplicate sids in a workflow.
    """
    sids = []
    def get_sids_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            if 'sid' in step:
                sids.append(step['sid'])
            if step.get('step_type') == 'for_loop' and 'steps' in step:
                get_sids_recursive(step['steps'])

    get_sids_recursive(steps)
    
    seen = set()
    duplicates = set()
    for sid in sids:
        if sid in seen:
            duplicates.add(sid)
        else:
            seen.add(sid)
    
    if duplicates:
        print(f"  [Info] Found {len(duplicates)} duplicate SIDs: {', '.join(sorted(list(duplicates)))}")
    return list(duplicates)

def find_broken_links(steps):
    """
    Detects steps in a workflow where 'next_sid', 'next_sid_if_true',
    or 'next_sid_if_false' does not correspond to an existing 'sid'.
    """
    all_sids = get_all_sids(steps)
    # "end" is a valid terminal state.
    all_sids.add("end")
    broken_links = []

    def check_step_links_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            current_sid = step.get('sid', 'N/A')

            # Check next_sid for action and other step types. Empty is OK.
            if 'next_sid' in step and step['next_sid'] and step['next_sid'] not in all_sids:
                broken_links.append({
                    "from_sid": current_sid,
                    "to_sid": step['next_sid'],
                    "link_type": "next_sid"
                })

            # Check branch-specific sids. Empty is NOT OK.
            if step.get('step_type') == 'branch':
                if not step.get('next_sid_if_true') or step.get('next_sid_if_true') not in all_sids:
                    broken_links.append({
                        "from_sid": current_sid,
                        "to_sid": step.get('next_sid_if_true', 'MISSING'),
                        "link_type": "next_sid_if_true"
                    })
                if not step.get('next_sid_if_false') or step.get('next_sid_if_false') not in all_sids:
                    broken_links.append({
                        "from_sid": current_sid,
                        "to_sid": step.get('next_sid_if_false', 'MISSING'),
                        "link_type": "next_sid_if_false"
                    })
            
            # Recurse into for_loops
            if step.get('step_type') == 'for_loop' and 'steps' in step:
                check_step_links_recursive(step['steps'])

    check_step_links_recursive(steps)
    if broken_links:
        print(f"  [Info] Found {len(broken_links)} broken links.")
    return broken_links

def verify_step_types(steps):
    """
    Verifies that all steps in the workflow have a valid `step_type`.
    """
    allowed_step_types = {'action', 'branch', 'for_loop', 'break'}
    invalid_steps = []

    def check_step_types_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            step_type = step.get('step_type')
            if step_type not in allowed_step_types:
                invalid_steps.append({
                    "sid": step.get('sid', 'N/A'),
                    "step_type": step_type
                })
            
            if step_type == 'for_loop' and 'steps' in step:
                check_step_types_recursive(step['steps'])

    check_step_types_recursive(steps)
    if invalid_steps:
        print(f"  [Info] Found {len(invalid_steps)} steps with invalid step_types.")
    return invalid_steps

def verify_collection_values(steps):
    """
    Verifies that all for_loop steps in the workflow have a valid `collection` value.
    """
    allowed_collections = {"singleton", "all_locations", "all_closed_containers", "all_opened_containers"}
    invalid_steps = []

    def check_collection_values_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            if step.get('step_type') == 'for_loop':
                collection_value = step.get('collection')
                if collection_value not in allowed_collections:
                    invalid_steps.append({
                        "sid": step.get('sid', 'N/A'),
                        "collection": collection_value
                    })
                # Recursively check nested steps
                if 'steps' in step:
                    check_collection_values_recursive(step['steps'])

    check_collection_values_recursive(steps)
    if invalid_steps:
        print(f"  [Info] Found {len(invalid_steps)} for_loop steps with invalid collection values.")
    return invalid_steps

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
