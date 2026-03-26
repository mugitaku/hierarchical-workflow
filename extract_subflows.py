import json
import argparse
import sys
import os
from lib.llm_api import initialize_router, completion_with_backoff
from lib.db_operations import initialize_db, get_all_subflows, register_subflow_in_db
from lib.utils import (
    extract_json,
    find_duplicate_sids,
    find_broken_links,
    find_unreachable_steps,
    verify_step_types,
    verify_collection_values,
    detect_missing_keys,
    check_after_llm
)

def extract_subflows_llm(workflow_steps, args, router):
    """
    Uses an LLM to extract sub-workflows from a workflow in two steps.
    """
    # --- Step 1: Decompose and Name ---
    
    # Create a detailed prompt for the first request
    prompt1 = f"""<INSTRUCTIONS>
Your primary task is to extract reusable sub-workflows from the original workflow shown in INPUT_WORKFLOW section. 
A "sub-workflow" is a self-contained sequence of steps that achieves a specific part of the overall goal.

Follow these steps:
1. extract reusable sub-workflows from the original workflow
    * Each sub-workflow must contain multiple steps.
2. add a unique "name" key for each sub-workflow whose value describes its purpose as specific as possible. 
    * Each name MUST be between 4 and 15 words.
    * The base form of a verb must be used at the beginning of the name.
    * The base form of verb can only be used once in each name.
</INSTRUCTIONS>

<INPUT_WORKFLOW>
```json
{json.dumps(workflow_steps, indent=2)}
```
</INPUT_WORKFLOW>

<OUTPUT_FORMAT>
* Do not include any other text or explanations outside of the JSON output.
* Each sub-workflow has "name" and "steps" keys.
</OUTPUT_FORMAT>
"""

    try:
        response1 = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": prompt1}],
            temperature=0.0,
            router=router
        )
        
        response_content1 = response1['choices'][0]['message']['content']
        print("■response_content1:", response_content1)
        
        # Extract the JSON part of the response
        sub_workflows_named_json = extract_json(response_content1)
        
        if not (sub_workflows_named_json and isinstance(sub_workflows_named_json, list)):
            print("Error: LLM did not return a valid list of sub-workflows in step 1.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"An error occurred during step 1 (Decompose and Name): {e}", file=sys.stderr)
        return None

    # --- Step 2: Abstract ---

    # Create a detailed prompt for the second request
    prompt2 = f"""<INSTRUCTIONS>
Your primary task is to abstract all the object names in the value of "name", "sid", "next_sid", "next_sid_if_true", "next_sid_if_false", "action" keys.
For example, you can abstract the object names to "object", "container", "heater", "refrigeration appliance", "room", etc
</INSTRUCTIONS>

<INPUT_WORKFLOWS>
```json
{json.dumps(sub_workflows_named_json, indent=2)}
```
</INPUT_WORKFLOWS>

<CONSTRAINTS>
* Keep the structure of the workflow intact
* DO NOT output thinking process
* The base form of a verb must be used at the beginning of the value of each name key
</CONSTRAINTS>


"""
    try:
        response2 = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": prompt2}],
            temperature=0.0,
            router=router
        )
        
        response_content2 = response2['choices'][0]['message']['content']
        print("■response_content2:", response_content2)
        
        # Extract the JSON part of the response
        sub_workflows_abstracted_json = extract_json(response_content2)
        
        if sub_workflows_abstracted_json and isinstance(sub_workflows_abstracted_json, list):
            return sub_workflows_abstracted_json
        else:
            print("Error: LLM did not return a valid list of sub-workflows in step 2.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"An error occurred during step 2 (Abstract): {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sub-workflows from a workflow JSON file using an LLM.')
    parser.add_argument('--file_path', help='Path to the workflow JSON file.')
    parser.add_argument('--model', type=str, default='openrouter/google/gemma-3-27b-it:free', help='The model to use for extraction.')
    parser.add_argument("--actions_file", type=str, default="prompts/known_actions-simple.json", help="Action definition file path")
    args = parser.parse_args()

    try:
        with open(args.file_path, 'r') as f:
            workflow_data = json.load(f)
        
        steps = workflow_data.get("steps", [])
        if not steps and isinstance(workflow_data, list):
            steps = workflow_data

        check_after_llm(steps)

        # --- Sub-workflow Extraction ---
        print("\n--- Extracting Sub-workflows ---")
        router = initialize_router()
        sub_workflows = extract_subflows_llm(steps, args, router)

        if sub_workflows:
            # --- Database and Duplicate Check ---
            model, collection = initialize_db(disable_db=False)
            
            existing_subflow_docs = get_all_subflows(collection)
            existing_subflows = []
            for doc in existing_subflow_docs:
                try:
                    existing_subflows.append(json.loads(doc))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse a document from the database.", file=sys.stderr)

            # Create the subflows directory if it doesn't exist
            if not os.path.exists('subflows'):
                os.makedirs('subflows')

            for sub_workflow in sub_workflows:
                is_duplicate = False
                for existing_workflow in existing_subflows:
                    if sub_workflow.get('steps') == existing_workflow.get('steps'):
                        is_duplicate = True
                        print(f"Sub-workflow '{sub_workflow.get('name')}' is a duplicate of an existing sub-workflow in DB. Skipping.")
                        break
                
                if not is_duplicate:
                    # --- Save to File ---
                    name = sub_workflow.get("name", "unnamed_sub_workflow")
                    base_filename = name.lower().replace(" ", "_")
                    filename = base_filename + ".json"
                    filepath = os.path.join('subflows', filename)

                    counter = 1
                    while os.path.exists(filepath):
                        filename = f"{base_filename}_{counter}.json"
                        filepath = os.path.join('subflows', filename)
                        counter += 1
                    
                    with open(filepath, 'w') as f:
                        json.dump(sub_workflow, f, indent=2)
                    print(f"Saved new sub-workflow '{name}' to {filepath}")

                    # --- Register in DB ---
                    register_subflow_in_db(collection, model, sub_workflow)

        else:
            print("Failed to extract sub-workflows.", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File not found at {args.file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
