import os
import sys
import json
import subprocess
from datetime import datetime
import copy

from lib.config_loader import parse_arguments, load_file_content, get_action_definitions, load_and_format_actions
from lib.utils import find_broken_links, normalize_workflow_steps, find_duplicate_sids, verify_step_types, verify_collection_values
from lib.llm_api import initialize_router
from lib.db_operations import initialize_db
from lib.workflow_logic import generate_workflow
from lib.workflow_decomposition import sub_workflow_recursive
from lib.workflow_refinement import refine_workflow_content, refine_workflow_format

def main():
    args = parse_arguments()

    # 1. アクション定義の取得 (辞書リスト全体を取得)
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
    
    def generate_diagram_for_file(filename, title):
        """Generates a diagram for a given workflow file."""
        if not args.generate_diagram:
            return
        
        print(f"\n--- Generating diagram for {title} workflow ---")
        diagram_script = "lib/diagram.py"
        
        if os.path.exists(diagram_script):
            try:
                # 同じPythonインタプリタを使用してdiagram.pyを実行
                cmd = [sys.executable, diagram_script, filename]
                print(f"Running: {' '.join(cmd)}")
                
                subprocess.run(cmd, check=True)
                print(f"✅ {title} のダイアグラム生成が完了しました。")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ {title} の diagram.py の実行中にエラーが発生しました: {e}")
            except Exception as e:
                print(f"❌ {title} のダイアグラム生成中に予期せぬエラーが発生しました: {e}")
        else:
            print(f"❌ '{diagram_script}' が現在のディレクトリに見つかりません。")

    # 3. 初期ワークフロー生成
    print("=== Phase 1: 初期ワークフロー生成 ===")

    if args.disable_subflow:
        limited = True
        types = ['primitive1', 'primitive2', "milestone"]
    elif args.disable_db:
        limited = False
        types = ['primitive1', 'primitive2', "milestone"]
    else:
        limited = False
        types = ['primitive1', 'primitive2', 'complex', "milestone"]
    
    user_prompt_add_main = f"""
    <INSTRUCTIONS>
    {load_and_format_actions(args.actions_file, limited, types)}
    </INSTRUCTIONS>
    """

    initial_workflow_obj = generate_workflow(sys_prompt_main, user_prompt_origin, user_prompt_add_main, format_content, args, router, local_embed_model, collection)
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
    
    initial_filename = os.path.join(output_dir, f"{sanitized_model_name}-1-initial-{timestamp}.txt")
    write_workflow_to_txt(initial_filename, initial_workflow_wrapped, "Initial Workflow")
    generate_diagram_for_file(initial_filename, "Initial")
    
    if args.disable_subflow:
        final_steps = initial_steps
        pre_refined_final_steps = initial_steps
    else:
        print("\n=== Phase 2: アクション照合とサブフロー生成 ===")
        if args.disable_example:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content
        else:
            sys_prompt_sub = load_file_content(args.sys_sub_file) + "\n" + format_content + "\n" + example_content
        
        pre_refined_final_steps = sub_workflow_recursive(user_prompt_env, copy.deepcopy(initial_steps), action_definitions, sys_prompt_sub, args, router, local_embed_model, collection)
        final_steps = pre_refined_final_steps

    pre_refined_workflow = {
        "sid": "root",
        "step_type": "for_loop",
        "iterator": "_",
        "collection": "singleton",
        "steps": pre_refined_final_steps
    }

    pre_refine_filename = os.path.join(output_dir, f"{sanitized_model_name}-2-pre-refine-{timestamp}.txt")
    write_workflow_to_txt(pre_refine_filename, pre_refined_workflow, "Pre-Refined Final Workflow")
    generate_diagram_for_file(pre_refine_filename, "Pre-Refined")

    final_steps = refine_workflow_content(final_steps, user_prompt_origin, format_content, args, router)
    
    # Format refinement is now called without passing broken_links
    final_steps = refine_workflow_format(final_steps, user_prompt_origin, user_prompt_add_main, format_content, args, router)

    # ルートオブジェクトで最終的なワークフローをラップする
    final_workflow = {
        "sid": "root",
        "step_type": "for_loop",
        "iterator": "_",
        "collection": "singleton",
        "steps": final_steps
    }

    final_filename = os.path.join(output_dir, f"{sanitized_model_name}-3-final-{timestamp}.txt")
    write_workflow_to_txt(final_filename, final_workflow, "Final Workflow")
    generate_diagram_for_file(final_filename, "Final")


    # --- Final Validation ---
    print("\n" + "="*30)
    print("VALIDATING FINAL WORKFLOW")
    print("="*30)
    final_steps = final_workflow.get("steps", [])
    broken_links = find_broken_links(final_steps)
    duplicate_sids = find_duplicate_sids(final_steps)

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
    
    # --- 結果出力 ---
    print("\n" + "="*30)
    print("FINAL WORKFLOW JSON")
    print("="*30)
    final_output_str = json.dumps(final_workflow, indent=2, ensure_ascii=False)
    print(final_output_str)

if __name__ == "__main__":
    main()
