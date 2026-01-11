import uuid
from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps
from lib.config_loader import load_file_content, load_and_format_actions
from lib.db_operations import search_subflows
from lib.workflow_refinement import refine_workflow_format

def generate_workflow(sys_prompt, user_prompt_origin, user_prompt_add, format_content, args, router, local_embed_model, collection):
    print(f"\n>>> 生成プロセス開始")
    # Step 1. Task Decomposition
    reference_info = ""
    # primitive1 is routine action, so it is not used in subtask decomposition.
    if args.disable_db:
        limited=True
        types=['primitive2']
    else:
        limited=True
        types=['primitive2', 'complex']

    action_list_main = load_and_format_actions(args.actions_file, limited, types)
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
    * {action_list_main}
    </CONSTRAINTS>
    <OUTPUT_FORMAT>
    subtask1, subtask2, ...
    </OUTPUT_FORMAT>
    """
    
    try:
        decomp_resp = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": decomp_prompt}],
            temperature=args.temperature,
            router=router
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
        found_subflows, reference_info = search_subflows(collection, local_embed_model, subtasks, args.disable_db)
    else:
        found_subflows, reference_info = [], ""

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
            temperature=args.temperature,
            router=router
        )
        print("■sys_prompt:", sys_prompt)
        print("■user_prompt:", user_prompt)
        
        response_content = response['choices'][0]['message']['content']
        print("■response_content:", response_content)

        data = extract_json(response_content)
        if not data:
            return None

        steps = normalize_workflow_steps(data)
        
        # Perform initial format refinement
        print("\n--- Initial Format Refinement ---")
        refined_steps = refine_workflow_format(steps, user_prompt_origin, user_prompt_add, format_content, args, router)
        
        return refined_steps

    except Exception as e:
        print(f"  [Error] ワークフロー生成API呼び出し中にエラーが発生しました: {e}")
        return None


