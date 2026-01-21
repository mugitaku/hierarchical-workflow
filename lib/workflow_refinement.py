import json
from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps, find_duplicate_sids, find_broken_links, find_unreachable_steps, verify_step_types, verify_collection_values, detect_missing_keys
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


def refine_workflow_flexibility(steps, user_prompt_origin, format_content, args, router):
    print("\n=== Phase 3a: Flexibility Refinement ===")
    
    workflow_json_str = json.dumps(steps, indent=2, ensure_ascii=False)
    action_list_refine_content = load_and_format_actions(args.actions_file, True, ['primitive1', 'primitive2', "milestone"])

    refine_prompt = f"""
<INSTRUCTIONS>
Your primary goal is to make the workflow shown in WORKFLOW section to be more flexible.
The workflow is to complete the task shown in TASK section.

Follow these instructions:
* {action_list_refine_content}
* Some objects in the environment might be unavailable due to damage or shortage, but your workflow must be executable in any situation.
To ensure executability, workflow must have multiple alternate paths. If you find multiple ways to complete your task, include them using conditional branches.
Here is an example of conditional branch for executability:
{{"sid": "availability_machine_1", "step_type": "branch", "condition": "machine 1 is available", "next_sid_if_true": "machine_1", "next_sid_if_false": "machine_2"}},
{{"sid": "machine_1", "step_type": "action", "action": "use machine 1", "next_sid": "pick_up_product"}},
{{"sid": "machine_2", "step_type": "action", "action": "use machine 2", "next_sid": "pick_up_product"}}
* Insert missing steps
* If the milestone actions are specified in task steps, the workflow MUST include them so any routes path the actions
</INSTRUCTIONS>

{user_prompt_origin}
{format_content}

<WORKFLOW>
{workflow_json_str}
</WORKFLOW>
"""
    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response (Flexibility):", response_content)

        refined_data = extract_json(response_content)
        if refined_data:
            return normalize_workflow_steps(refined_data)
        else:
            print("  [Warning] Flexibility refinement failed. Using original workflow.")
            return steps

    except Exception as e:
        print(f"  [Error] An error occurred during flexibility refinement: {e}")
        return steps

def refine_workflow_details(steps, user_prompt_origin, format_content, args, router):
    print("\n=== Phase 3b: Details Refinement ===")
    
    workflow_json_str = json.dumps(steps, indent=2, ensure_ascii=False)
    action_list_refine_content = load_and_format_actions(args.actions_file, True, ['primitive1', 'primitive2', "milestone"])

    refine_prompt = f"""
<INSTRUCTIONS>
Your primary goal is to make the workflow shown in WORKFLOW section to be executable.
The workflow is to complete the task shown in TASK section

Follow these instructions:
* Do not simplify the workflow
* Adapt the object names in the steps to your environment written in CONTEXT section
* If the agent needs to see objects at a location, he must go to the location beforehand
* {action_list_refine_content}
* If task steps are specified in TASK section, the steps in the workflow MUST follow the order.
    If the workflow has task steps out of order, fix them.
* An action step can contain only one action. If an action step contains multiple actions, split it into multiple steps
</INSTRUCTIONS>

{user_prompt_origin}
{format_content}

<WORKFLOW>
{workflow_json_str}
</WORKFLOW>
"""

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response (Details):", response_content)

        refined_data = extract_json(response_content)
        if refined_data:
            return normalize_workflow_steps(refined_data)
        else:
            print("  [Warning] Details refinement failed. Using original workflow.")
            return steps

    except Exception as e:
        print(f"  [Error] An error occurred during details refinement: {e}")
        return steps


def refine_workflow_format(steps, user_prompt_origin, action_limited, action_types, format_content, args, router):
    print("\n=== Format Refinement ===")
    
    # --- Inner function for validation ---
    def validate_workflow(steps_to_validate):
        duplicate_sids = find_duplicate_sids(steps_to_validate)
        broken_links = find_broken_links(steps_to_validate)
        unreachable_sids = find_unreachable_steps(steps_to_validate)
        invalid_step_types = verify_step_types(steps_to_validate)
        invalid_collections = verify_collection_values(steps_to_validate)
        missing_keys = detect_missing_keys(steps_to_validate)
        
        action_definitions = get_action_definitions(args.actions_file)
        invalid_actions = verify_actions(steps_to_validate, action_definitions)

        errors = {
            "duplicate_sids": duplicate_sids,
            "broken_links": broken_links,
            "unreachable_sids": unreachable_sids,
            "invalid_step_types": invalid_step_types,
            "invalid_collections": invalid_collections,
            "invalid_actions": invalid_actions,
            "missing_keys": missing_keys
        }
        has_errors = any(bool(v) for v in errors.values())
        return errors, has_errors

    # --- Initial Validation ---
    initial_errors, has_initial_errors = validate_workflow(steps)
    
    if not has_initial_errors:
        print("  [Info] Workflow format is valid. Skipping format refinement.")
        return steps

    # --- Error Reporting and Prompt Generation ---
    error_instructions = []
    if initial_errors["duplicate_sids"]:
        error_instructions.append(
            f"* The following duplicate sids were detected. You MUST fix them by assigning new unique sids:\n" +
            "\n".join([f"  - {sid}" for sid in sorted(initial_errors['duplicate_sids'])])
        )
    if initial_errors["broken_links"]:
        error_instructions.append(
            f"* The following broken links were detected. You MUST fix them:\n" +
            "\n".join([f"  - from: {link['from_sid']}, to: {link['to_sid']}, type: {link['link_type']}" for link in initial_errors['broken_links']])
        )
    if initial_errors["unreachable_sids"]:
        error_instructions.append(
            f"* The following sids are unreachable. You MUST fix them by either connecting them to the workflow or removing them:\n" +
            "\n".join([f"  - {sid}" for sid in sorted(initial_errors['unreachable_sids'])])
        )
    if initial_errors["invalid_step_types"]:
        error_instructions.append(
            f"* The following steps have invalid `step_type`. You MUST fix them (allowed types are: action, branch, for_loop, break):\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_type: '{s['step_type']}'" for s in initial_errors['invalid_step_types']])
        )
    if initial_errors["invalid_collections"]:
        error_instructions.append(
            f"* The following for_loop steps have invalid `collection` values. You MUST fix them (allowed values are: all_locations, all_closed_containers, all_opened_containers):\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_collection: '{s['collection']}'" for s in initial_errors['invalid_collections']])
        )
    if initial_errors["invalid_actions"]:
        error_instructions.append(
            f"* The following steps have invalid actions. You MUST fix them to use actions from the allowed list:\n" +
            "\n".join([f"  - sid: {s['sid']}, invalid_action: '{s['action']}'" for s in initial_errors['invalid_actions']])
        )
    if initial_errors["missing_keys"]:
        error_instructions.append(
            f"* The following steps are missing required keys for their `step_type`. You MUST add the missing keys:\n" +
            "\n".join([f"  - sid: {s['sid']}, step_type: {s['step_type']}, missing_keys: {s['missing_keys']}" for s in initial_errors['missing_keys']])
        )

    error_prompt_section = "\n".join(error_instructions)
    intermediate_json_str = json.dumps(steps, indent=2, ensure_ascii=False)

    refine_prompt = f"""
<INSTRUCTIONS>
The workflow in WORKFLOW section has formatting errors. You MUST fix them based on the following error report.
The overall structure MUST be preserved.
When fixing the workflow, ensure it still achieves the original goal described in the TASK section.

{load_and_format_actions(args.actions_file, action_limited, action_types)}
{user_prompt_origin}
{format_content}
<ERROR_REPORT>
{error_prompt_section}
</ERROR_REPORT>
</INSTRUCTIONS>
<WORKFLOW>
{intermediate_json_str}
</WORKFLOW>
"""
    print("■refine_prompt:", refine_prompt)

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        print("LLM Refinement response (Format):", response_content)

        refined_data = extract_json(response_content)
        if not refined_data:
            print("  [Warning] Format refinement LLM call failed to produce valid JSON. Returning original workflow with errors.")
            return steps
        
        refined_steps = normalize_workflow_steps(refined_data)

        # --- Post-Refinement Validation ---
        final_errors, has_final_errors = validate_workflow(refined_steps)
        if has_final_errors:
            print("\n  [Warning] Automated format refinement was UNSUCCESSFUL. The workflow still contains errors.")
            # Optionally print the remaining errors for debugging
            # for error_type, error_list in final_errors.items():
            #     if error_list:
            #         print(f"    - Remaining '{error_type}': {error_list}")
        else:
            print("\n  [Info] Automated format refinement was successful.")

        return refined_steps

    except Exception as e:
        print(f"  [Error] An error occurred during format refinement: {e}")
        return steps

