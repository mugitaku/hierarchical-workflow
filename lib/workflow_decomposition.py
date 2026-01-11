import uuid
import copy
from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps, find_broken_links, get_all_sids
from lib.config_loader import load_file_content, load_and_format_actions
from lib.workflow_logic import generate_workflow

def is_decomposable(action_str, action_definitions):
    """
    アクション文字列を定義リストと照合し、分解可能かどうかを判断する。
    最も長くマッチしたアクション定義を探し、そのtypeが'complex'か、または定義が見つからなければTrueを返す。
    typeが'primitive1', 'primitive2', 'milestone'であればFalseを返す。
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
        # 'complex' is decomposable. 'primitive1', 'primitive2', and 'milestone' are not.
        return action_type == 'complex'

    # No definition found, so we assume it's a complex action that needs decomposition.
    return True

def sub_workflow_recursive(user_prompt_origin, steps_list, action_definitions, sys_prompt_sub, args, router, local_embed_model, collection, current_depth=0):

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
                    limited=True
                    types=['primitive1', 'primitive2']
                elif args.disable_db:
                    limited=False
                    types=['primitive1', 'primitive2']
                else:
                    limited=True
                    types=['primitive1', 'primitive2', 'complex']
                action_list_sub = load_and_format_actions(args.actions_file, limited, types)
                user_prompt_add_sub =f"""
<TASK>
Generate workflow from "{action_content}" action if possible,
 but DO NOT use "{action_content}" itself in the workflow.
The object names and action names in the workflow must be adapted to your task.
{action_list_sub}
</TASK>
                """
                
                sid_to_replace = step.get("sid")
                next_sid_to_replace = step.get("next_sid")

                # 1. Generate raw sub-workflow without stitching
                sub_workflow_data = generate_workflow(sys_prompt_sub, user_prompt_origin, user_prompt_add_sub, None, args, router, local_embed_model, collection)              
                
                sub_steps = normalize_workflow_steps(sub_workflow_data)

                if sub_steps:
                    # 2. Recursively generate the new sub-workflow
                    generated_sub_steps = sub_workflow_recursive(
                        user_prompt_origin,
                        sub_steps, 
                        action_definitions,
                        sys_prompt_sub,
                        args,
                        router,
                        local_embed_model,
                        collection,
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
                        all_sids_in_subtree = get_all_sids(generated_sub_steps)
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
                    sys_prompt_sub,
                    args,
                    router,
                    local_embed_model,
                    collection,
                    current_depth
                )
            new_steps.append(step)
            
        else:
            new_steps.append(step)

    return new_steps
