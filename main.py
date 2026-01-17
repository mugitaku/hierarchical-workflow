import os
import sys
import json
import subprocess
from datetime import datetime
import copy

from lib.config_loader import parse_arguments, load_file_content, get_action_definitions, load_and_format_actions
from lib.utils import (
    find_broken_links, 
    normalize_workflow_steps, 
    find_duplicate_sids, 
    verify_step_types, 
    verify_collection_values, 
    find_unreachable_steps, 
    wrap_workflow_with_root,
    write_workflow_to_txt,
    generate_diagram_for_file
)
from lib.llm_api import initialize_router
from lib.db_operations import initialize_db
from lib.workflow_logic import generate_workflow
from lib.workflow_decomposition import sub_workflow_recursive
from lib.workflow_refinement import refine_workflow_flexibility, refine_workflow_details, refine_workflow_format

def main():
    args = parse_arguments()

    # 1. Get action definitions (gets the entire list of dictionaries)
    action_definitions = get_action_definitions(args.actions_file)

    # 2. prompts
    format_content = load_file_content(args.format_file)
    example_content = load_file_content(args.example_file)
    if args.disable_example:
        sys_prompt_main = load_file_content(args.sys_main_file) + "\n" + format_content
    else:
        sys_prompt_main = load_file_content(args.sys_main_file) + "\n" + format_content + "\n" + example_content
    user_prompt_env = f"""<CONTEXT>{load_file_content(args.user_env_file)}</CONTEXT>"""
    user_prompt_origin = f"""<CONTEXT>{load_file_content(args.user_env_file)}</CONTEXT><TASK>{load_file_content(args.user_task_file)}</TASK>"""

    # --- Router & DB Initialization ---
    router = initialize_router()
    local_embed_model, collection = initialize_db(args.disable_db)

    # --- Output Setup ---
    env_file_basename = os.path.splitext(os.path.basename(args.user_env_file))[0] if args.user_env_file else "default_env"
    task_file_basename = os.path.splitext(os.path.basename(args.user_task_file))[0] if args.user_task_file else "default_task"
    output_dir_name = f"env_{env_file_basename}_task_{task_file_basename}"
    output_dir = os.path.join("output", output_dir_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    sanitized_model_name = args.model.split(':')[0].split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # 3. Initial Workflow Generation
    print("=== Phase 1: Initial Workflow Generation ===")

    if args.disable_subflow:
        action_limited = True
        action_types = ['primitive1', 'primitive2', "milestone"]
    elif args.disable_db:
        action_limited = False
        action_types = ['primitive1', 'primitive2', "milestone"]
    else:
        action_limited = False
        action_types = ['primitive1', 'primitive2', 'complex', "milestone"]

    initial_workflow_obj = generate_workflow(sys_prompt_main, user_prompt_origin, action_limited, action_types, format_content, args, router, local_embed_model, collection)
    initial_steps = normalize_workflow_steps(initial_workflow_obj)

    if not initial_steps:
        print("Initial generation failed. Exiting.")
        exit(1)

    initial_workflow_wrapped = wrap_workflow_with_root(initial_steps)
    
    initial_filename = os.path.join(output_dir, f"{sanitized_model_name}-1-initial-{timestamp}.txt")
    write_workflow_to_txt(initial_filename, initial_workflow_wrapped, "Initial Workflow", args)
    generate_diagram_for_file(initial_filename, "Initial", args)
    
    if args.disable_subflow:
        final_steps = initial_steps
        pre_refined_final_steps = initial_steps
    else:
        print("\n=== Phase 2: Action Matching and Subflow Generation ===")
        if args.disable_example:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content
        else:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content + "\n" + example_content
        
        pre_refined_final_steps = sub_workflow_recursive(user_prompt_env, copy.deepcopy(initial_steps), action_definitions, sys_prompt_sub, args, router, local_embed_model, collection)
        final_steps = pre_refined_final_steps

    pre_refined_workflow = wrap_workflow_with_root(pre_refined_final_steps)

    pre_refine_filename = os.path.join(output_dir, f"{sanitized_model_name}-2-pre-refine-{timestamp}.txt")
    write_workflow_to_txt(pre_refine_filename, pre_refined_workflow, "Pre-Refined Final Workflow", args)
    generate_diagram_for_file(pre_refine_filename, "Pre-Refined", args)

    final_steps = refine_workflow_flexibility(final_steps, user_prompt_origin, format_content, args, router)

    # Generate diagram after flexibility refinement
    post_flex_workflow = wrap_workflow_with_root(final_steps)
    post_flex_filename = os.path.join(output_dir, f"{sanitized_model_name}-3-post-flex-{timestamp}.txt")
    write_workflow_to_txt(post_flex_filename, post_flex_workflow, "Post-Flexibility-Refined Workflow", args)
    generate_diagram_for_file(post_flex_filename, "Post-Flexibility-Refined", args)

    final_steps = refine_workflow_details(final_steps, user_prompt_origin, format_content, args, router)
    
    # Generate diagram after details refinement
    post_details_workflow = wrap_workflow_with_root(final_steps)
    post_details_filename = os.path.join(output_dir, f"{sanitized_model_name}-4-post-details-{timestamp}.txt")
    write_workflow_to_txt(post_details_filename, post_details_workflow, "Post-Details-Refined Workflow", args)
    generate_diagram_for_file(post_details_filename, "Post-Details-Refined", args)

    # Format refinement is now called without passing broken_links
    final_steps = refine_workflow_format(final_steps, user_prompt_origin, action_limited, action_types, format_content, args, router)

    # Wrap the final workflow in a root object
    final_workflow = wrap_workflow_with_root(final_steps)

    final_filename = os.path.join(output_dir, f"{sanitized_model_name}-5-final-{timestamp}.txt")
    write_workflow_to_txt(final_filename, final_workflow, "Final Workflow", args)
    generate_diagram_for_file(final_filename, "Final", args)


    # --- Final Validation ---
    print("\n" + "="*30)
    print("VALIDATING FINAL WORKFLOW")
    print("="*30)
    final_steps = final_workflow.get("steps", [])
    broken_links = find_broken_links(final_steps)
    duplicate_sids = find_duplicate_sids(final_steps)
    unreachable_sids = find_unreachable_steps(final_steps)

    if broken_links:
        print("❌ Found broken links in the final workflow:")
        for link in broken_links:
            print(f"  - From SID '{link['from_sid']}' to non-existent SID '{link['to_sid']}' (link type: {link['link_type']})")
    else:
        print("✅ Final workflow validation successful. No broken links found.")

    if duplicate_sids:
        print(f"❌ Found {len(duplicate_sids)} duplicate SIDs in the final workflow:")
        for sid in duplicate_sids:
            print(f"  - SID '{sid}' is duplicated.")
    else:
        print("✅ Final workflow validation successful. No duplicate SIDs found.")

    if unreachable_sids:
        print(f"❌ Found {len(unreachable_sids)} unreachable SIDs in the final workflow:")
        for sid in unreachable_sids:
            print(f"  - SID '{sid}' is unreachable.")
    else:
        print("✅ Final workflow validation successful. No unreachable SIDs found.")

    invalid_step_types = verify_step_types(final_steps)
    if invalid_step_types:
        print(f"❌ Found {len(invalid_step_types)} steps with invalid step_types in the final workflow:")
        for step in invalid_step_types:
            print(f"  - SID '{step['sid']}' has an invalid step_type: '{step['step_type']}'")
    else:
        print("✅ Final workflow validation successful. No invalid step_types found.")

    invalid_collections = verify_collection_values(final_steps)
    if invalid_collections:
        print(f"❌ Found {len(invalid_collections)} for_loop steps with invalid collection values in the final workflow:")
        for step in invalid_collections:
            print(f"  - SID '{step['sid']}' has an invalid collection value: '{step['collection']}'")
    else:
        print("✅ Final workflow validation successful. No invalid collection values found.")
    
    # --- Output Results ---
    print("\n" + "="*30)
    print("FINAL WORKFLOW JSON")
    print("="*30)
    final_output_str = json.dumps(final_workflow, indent=2, ensure_ascii=False)
    print(final_output_str)

if __name__ == "__main__":
    main()

