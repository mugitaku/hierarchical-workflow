import json
from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps, find_duplicate_sids
from lib.config_loader import load_and_format_actions

def refine_workflow_content(steps, user_prompt_origin, format_content, args, router):
    """
    Refines the content of the workflow by inserting missing steps, adding conditional branches,
    and making actions more specific.
    """
    # Step 1: Content Refinement (内容の改善)
    print("\n=== Phase 3: 最終ワークフローの最適化 (Step 1: Content) ===")
    
    workflow_json_str = json.dumps(steps, indent=2, ensure_ascii=False)
    known_actions = load_and_format_actions(args.actions_file, types=['primitive1', 'primitive2', "milestone"])

    refine_prompt_1 = f"""
<INSTRUCTIONS>
Refine the workflow showed in WORKFLOW_TO_REFINE section.
The workflow is to complete the task showed in OVERALL_TASK section.

Follow these instructions to refine the workflow content:
* If task steps are specified in OVERALL_TASK section, you MUST follow the order
* If the milestone action is specified in task steps, you MUST use the action
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
* Only the following actions are allowed in the workflow. 
  If the milestone action is specified in task steps, you MUST use the action.
  {known_actions}
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


def refine_workflow_format(steps, format_content, broken_links, args, router):
    """
    Refines the format of the workflow by ensuring unique sids, consistent links,
    and splitting multi-action steps. It can also fix provided broken links.
    """
    # Step 2: Format Refinement (フォーマットの修正)
    print("\n=== Phase 3: 最終ワークフローの最適化 (Step 2: Format) ===")
    
    intermediate_json_str = json.dumps(steps, indent=2, ensure_ascii=False)

    broken_links_instruction = ""
    if broken_links:
        broken_links_str = "\n".join([f"  - from: {link['from_sid']}, to: {link['to_sid']}, type: {link['link_type']}" for link in broken_links])
        broken_links_instruction = f"""
* The following broken links were detected. You MUST fix them.
{broken_links_str}
"""

    duplicate_sids = find_duplicate_sids(steps)
    duplicate_sids_instruction = ""
    if duplicate_sids:
        duplicate_sids_str = "\n".join([f"  - {sid}" for sid in sorted(duplicate_sids)])
        duplicate_sids_instruction = f"""
* The following duplicate sids were detected. You MUST fix them by assigning new unique sids.
{duplicate_sids_str}
"""

    refine_prompt_2 = f"""
<INSTRUCTIONS>
Refine the workflow showed in WORKFLOW_TO_REFINE section.
Follow these instructions to fix the workflow format:
* Conform the workflow to output format
    * All steps have unique `sid`
    * All `sid` and `next_sid` links are consistent
    * If a action step contains multiple actions, split it into multiple steps
    * Allowed step_type = ['action', 'branch', "for_loop", "break"]
{broken_links_instruction}
{duplicate_sids_instruction}
</INSTRUCTIONS>

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
            print("  [Warning] Step 2 リファインに失敗しました。Step 1の結果(または元)を維持します。")
            return steps

        return normalize_workflow_steps(refined_data)

    except Exception as e:
        print(f"  [Error] Step 2 リファイン中にエラーが発生しました: {e}")
        return steps
