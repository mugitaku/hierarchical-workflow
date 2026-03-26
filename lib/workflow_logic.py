from lib.llm_api import completion_with_backoff
from lib.utils import extract_json, normalize_workflow_steps
from lib.config_loader import load_file_content, load_and_format_actions
from lib.db_operations import search_subflows
from lib.workflow_refinement import refine_workflow_format

def generate_workflow(sys_prompt, user_prompt_origin, action_limited, action_types, format_content, args, router, local_embed_model, collection):
    print(f"\n>>> Starting generation process")
    reference_info = ""
    subtasks = []
    if not args.disable_decomp: 
        # Step 1. Task Decomposition
        if args.disable_db:
            action_limited_decomp=True
            # primitive1 is routine action, so it is not used in subtask decomposition.
            action_types_decomp=['primitive2']
        else:
            #action_limited_decomp=True
            #action_types_decomp=['primitive2', 'complex']

            # instead of using complex actions, agents can use arbitrary actions
            action_limited_decomp=False
            action_types_decomp=['primitive2']

        decomp_prompt = f"""
        <INSTRUCTIONS>
        List major subtasks to complete the following task: "{user_prompt_origin}"
        </INSTRUCTIONS>
        <CONSTRAINTS>
        * Abstract all the object names of your subtasks 
            (e.g. "machine" is abstracted to "object", "cup" is abstracted to "container", "stove" is abstracted to "heater", etc)
        * Subtasks MUST be separated by commas
        * Subtasks MUST be 15 words or less
        * The number of subtasks MUST be between 20 and 30
        * DO NOT use numbering or newlines
        * DO NOT output thinking process
        * DO NOT output duplicate subtasks
        * {load_and_format_actions(args.actions_file, action_limited_decomp, action_types_decomp)}
        </CONSTRAINTS>
        <OUTPUT_FORMAT>
        subtask1, subtask2, ...
        </OUTPUT_FORMAT>
        """
        print("■decomp_prompt:", decomp_prompt)
        
        try:
            decomp_resp = completion_with_backoff(
                model=args.model,
                messages=[{"role": "user", "content": decomp_prompt}],
                temperature=args.temperature,
                router=router,
                max_tokens=4096
            )
            # Split by comma and deduplicate
            raw_subtasks_with_duplicates = decomp_resp['choices'][0]['message']['content'].split(',')
            seen = set()
            for task in raw_subtasks_with_duplicates:
                task = task.strip()
                if task and task not in seen:
                    seen.add(task)
                    subtasks.append(task)

        except Exception as e:
            print(f"  [Warning] An error occurred during task decomposition (skipping): {e}")
            subtasks = []

        # Step 2. DB Search
        if not args.disable_db:
            found_subflows, reference_info = search_subflows(collection, local_embed_model, subtasks, args.disable_db)
        else:
            found_subflows, reference_info = [], ""

    # Step 3. Main Generation
    if not args.disable_decomp: 
        user_prompt = f"""
{user_prompt_origin}
<INSTRUCTIONS>
Refer to the following subtask candidates and the steps of workflow candidates.
The object names are generalized, so adapt them to your task to make a specific and executable workflow.

If there are multiple objects in your environment that correspond to the generalized object name, include each path using those objects in your workflow to ensure executability.

subtask candidates: {subtasks}

workflow candidates:
{reference_info}

{load_and_format_actions(args.actions_file, action_limited, action_types)}
</INSTRUCTIONS>
    """
    else:
        user_prompt = f"""
{user_prompt_origin}
<INSTRUCTIONS>
{load_and_format_actions(args.actions_file, action_limited, action_types)}
</INSTRUCTIONS>
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
        refined_steps = refine_workflow_format(steps, user_prompt_origin, action_limited, action_types, format_content, args, router)
        
        return refined_steps

    except Exception as e:
        print(f"  [Error] An error occurred during the workflow generation API call: {e}")
        return None
