import json
from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps, find_duplicate_sids, find_broken_links, verify_step_types, verify_collection_values
from lib.config_loader import load_and_format_actions, get_action_definitions

def verify_actions(steps, allowed_actions):
    """
    Verifies that all 'action' steps in the workflow use actions from the allowed list.
    """
    invalid_actions = []
    allowed_action_names = {action['action'] for action in allowed_actions}

    def check_steps_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            if step.get('step_type') == 'action':
                action_name = step.get('action')
                if action_name and not any(action_name.startswith(allowed) for allowed in allowed_action_names):
                    invalid_actions.append(step)
            
            if step.get('step_type') == 'for_loop' and 'steps' in step:
                check_steps_recursive(step['steps'])

    check_steps_recursive(steps)
    return invalid_actions


def refine_workflow_content(steps, user_prompt_origin, format_content, args, router):
    """
    Refines the content of the workflow by inserting missing steps, adding conditional branches,
    and making actions more specific.
    """
    
    print("\n=== Phase 3: Content Refinement (内容の改善) ===")
    
    workflow_json_str = json.dumps(steps, indent=2, ensure_ascii=False)
    action_list_refine_content = load_and_format_actions(args.actions_file, True, ['primitive1', 'primitive2', "milestone"])

    refine_prompt_1 = f"""
<INSTRUCTIONS>
If there are errors in the workflow shown in WORKFLOW_TO_REFINE section, fix them.
The workflow is to complete the task shown in TASK section.

Follow these instructions:
* If task steps are specified in TASK section, you MUST follow the order
* Insert missing steps to be successfully executed
* If there are multiple ways to complete your task,
    include them to the workflow using conditional branches,
    because some objects in the environment can be unavailable and your task must be completed.
    Here is the structure example of conditional branch for availability:
    {{"sid": "availability_machine_1", "step_type": "branch", "condition": "machine 1 is available", "next_sid_if_true": "machine_1", "next_sid_if_false": "machine_2"}},
    {{"sid": "machine_1", "step_type": "action", "action": "use machine 1", "next_sid": "pick_up_product"}},
    {{"sid": "machine_2", "step_type": "action", "action": "use machine 2", "next_sid": "pick_up_product"}}
* Adapt the object names and variable names in the steps to your environment written in CONTEXT section
* Check the agent's location and use "go to <room>" action step properly
* {action_list_refine_content}
* If a action step contains multiple actions, split it into multiple steps
* If the milestone action is specified in task steps, you MUST use the action
</INSTRUCTIONS>

{user_prompt_origin}

{format_content}

<WORKFLOW_TO_REFINE>
{workflow_json_str}
</WORKFLOW_TO_REFINE>
"""

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[
                {"role": "user", "content": refine_prompt_1}
            ],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response (Step 1):", response_content)

        refined_data = extract_json(response_content)
        if refined_data:
            return normalize_workflow_steps(refined_data)
        else:
            print("  [Warning] Step 1 リファインに失敗しました。元のワークフローを使用します。")
            return steps

    except Exception as e:
        print(f"  [Error] Step 1 リファイン中にエラーが発生しました: {e}")
        return steps


def refine_workflow_format(steps, user_prompt_origin, user_prompt_add, format_content, args, router):
    """
    Refines the format of the workflow by ensuring unique sids, consistent links,
    and splitting multi-action steps. It can also fix provided broken links.
    """
    print("\n=== Format Refinement ===")
    
    # Verify workflow integrity
    def add_missing_next_sid_recursive(step_list):
        if not isinstance(step_list, list):
            return
        for step in step_list:
            if step.get('step_type') == 'action' and 'next_sid' not in step:
                step['next_sid'] = ""
            
            if step.get('step_type') == 'for_loop' and 'steps' in step:
                add_missing_next_sid_recursive(step['steps'])

    add_missing_next_sid_recursive(steps)

    duplicate_sids = find_duplicate_sids(steps)
    broken_links = find_broken_links(steps)
    invalid_step_types = verify_step_types(steps)
    invalid_collections = verify_collection_values(steps)
    
    action_definitions = get_action_definitions(args.actions_file)
    invalid_actions = verify_actions(steps, action_definitions)

    has_errors = duplicate_sids or broken_links or invalid_step_types or invalid_collections or invalid_actions
    
    if not has_errors:
        print("  [Info] ワークフローのフォーマットは正常です。Step 2のリファインをスキップします。")
        return steps

    # --- Error Reporting and Prompt Generation ---
    error_instructions = []
    if duplicate_sids:
        error_instructions.append(
            f"* The following duplicate sids were detected. You MUST fix them by assigning new unique sids:\n" +
            "\n".join([f"  - {sid}" for sid in sorted(duplicate_sids)])
        )
    if broken_links:
        error_instructions.append(
            f"* The following broken links were detected. You MUST fix them:\n" +
            "\n".join([f"  - from: {link['from_sid']}, to: {link['to_sid']}, type: {link['link_type']}" for link in broken_links])
        )
    if invalid_step_types:
        error_instructions.append(
            f"* The following steps have invalid `step_type`. You MUST fix them (allowed types are: action, branch, for_loop, break):\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_type: '{s['step_type']}'" for s in invalid_step_types])
        )
    if invalid_collections:
        error_instructions.append(
            f"* The following for_loop steps have invalid `collection` values. You MUST fix them (allowed values for `collection` are: singleton, all_locations, all_closed_containers, all_opened_containers):\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_collection: '{s['collection']}'" for s in invalid_collections])
        )

    if invalid_actions:
        error_instructions.append(
            f"* The following steps have invalid actions. You MUST fix them to use actions from the allowed list:\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_action: '{s['action']}'" for s in invalid_actions])
        )

    error_prompt_section = "\n".join(error_instructions)
    intermediate_json_str = json.dumps(steps, indent=2, ensure_ascii=False)

    refine_prompt_2 = f"""
<INSTRUCTIONS>
The workflow in WORKFLOW_TO_REFINE has formatting errors. You MUST fix them based on the following error report.
The JSON format and structure MUST be preserved.
When fixing the workflow, ensure it still achieves the original goal described in the TASK section.

<ERROR_REPORT>
{error_prompt_section}
</ERROR_REPORT>
</INSTRUCTIONS>
{user_prompt_origin}
{user_prompt_add}
{format_content}
<WORKFLOW_TO_REFINE>
{intermediate_json_str}
</WORKFLOW_TO_REFINE>
"""

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[
                {"role": "user", "content": refine_prompt_2}
            ],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response (Step 2):", response_content)

        refined_data = extract_json(response_content)
        if not refined_data:
            print("  [Warning] Step 2 リファインに失敗しました。エラーのあるワークフローを返します。")
            return steps

        return normalize_workflow_steps(refined_data)

    except Exception as e:
        print(f"  [Error] Step 2 リファイン中にエラーが発生しました: {e}")
        return steps

