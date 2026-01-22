import json
import argparse
import sys
from lib.llm_api import initialize_router, completion_with_backoff
from lib.utils import (
    extract_json,
    find_duplicate_sids,
    find_broken_links,
    find_unreachable_steps,
    verify_step_types,
    verify_collection_values,
    detect_missing_keys
)

def extract_subflows_llm(workflow_steps, args, router):
    """
    Uses an LLM to extract sub-workflows from a workflow.
    """
    example_input_workflow = [
        {
            "sid": "move_container_to_sink",
            "step_type": "action",
            "action": "move ${container} to sink",
            "next_sid": "activate_sink"
        },
        {
            "sid": "activate_sink",
            "step_type": "action",
            "action": "activate sink",
            "next_sid": "fill_container_with_water"
        },
        {
            "sid": "fill_container_with_water",
            "step_type": "subflow",
            "subflow_id": "fill_container_with_water",
            "next_sid": "deactivate_sink"
        },
        {
            "sid": "deactivate_sink",
            "step_type": "action",
            "action": "deactivate sink",
            "next_sid": "pick_up_container"
        },
        {
            "sid": "pick_up_container",
            "step_type": "action",
            "action": "pick up ${container}",
            "next_sid": "pour_container_into_flower_pot"
        },
        {
            "sid": "pour_container_into_flower_pot",
            "step_type": "action",
            "action": "pour ${container} into flower pot",
            "next_sid": "wait"
        },
        {
            "sid": "wait",
            "step_type": "action",
            "action": "wait",
            "next_sid": "look_at_flower_pot"
        },
        {
            "sid": "look_at_flower_pot",
            "step_type": "action",
            "action": "look at flower pot",
            "next_sid": ""
        }
    ]

    example_output = [
        {
            "name": "water plant in flower pot from container",
            "steps": [
                {
                    "sid": "fill_container_with_water",
                    "step_type": "action",
                    "action": "fill ${container} with water",
                    "next_sid": "pour_container_into_flower_pot"
                },
                {
                    "sid": "pour_container_into_flower_pot",
                    "step_type": "action",
                    "action": "pour ${container} into flower pot",
                    "next_sid": "wait"
                },
                {
                    "sid": "wait",
                    "step_type": "action",
                    "action": "wait",
                    "next_sid": "look_at_flower_pot"
                },
                {
                    "sid": "look_at_flower_pot",
                    "step_type": "action",
                    "action": "look at flower pot",
                    "next_sid": ""
                }
            ]
        },
        {
            "name": "fill container with water from sink",
            "steps": [
                {
                    "sid": "move_container_to_sink",
                    "step_type": "action",
                    "action": "move ${container} to sink",
                    "next_sid": "activate_sink"
                },
                {
                    "sid": "activate_sink",
                    "step_type": "action",
                    "action": "activate sink",
                    "next_sid": "wait"
                },
                {
                    "sid": "wait",
                    "step_type": "action",
                    "action": "wait",
                    "next_sid": "deactivate_sink"
                },
                {
                    "sid": "deactivate_sink",
                    "step_type": "action",
                    "action": "deactivate sink",
                    "next_sid": "pick_up_container"
                },
                {
                    "sid": "pick_up_container",
                    "step_type": "action",
                    "action": "pick up ${container}",
                    "next_sid": "pour_container_into_flower_pot"
                }
            ]
        }
    ]

    # Create a detailed prompt for the LLM
    prompt = f"""<INSTRUCTIONS>
Your task is to make reusable, general sub-workflows from the original workflow shown in INPUT_WORKFLOW section. 
A "sub-workflow" is a self-contained sequence of steps that achieves a specific part of the overall goal.
The original workflow should be reconstructible by combining these sub-workflows.

Follow these steps:
1. decompose the workflow into reusable sub-workflows
2. abstract all the object names of your sub-workflows
3. add a specific and detailed name for each sub-workflow that describes what and how the sub-workflow does. The name MUST be 15 words or less.
</INSTRUCTIONS>

<FEW_SHOT_EXAMPLE>
Here is an example of a input workflow and the desired output.

**Input Workflow:**
```json
{json.dumps(example_input_workflow, indent=2)}
```

**Desired Output:**
```json
{json.dumps(example_output, indent=2)}
```
</FEW_SHOT_EXAMPLE>

<INPUT_WORKFLOW>
```json
{json.dumps(workflow_steps, indent=2)}
```
</INPUT_WORKFLOW>

<OUTPUT_FORMAT>
Your output MUST be a valid JSON list of objects, where each object has a "name" and a "steps" list.
Do not include any other text or explanations outside of the JSON output.
</OUTPUT_FORMAT>
"""

    try:
        response = completion_with_backoff(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            router=router
        )
        
        response_content = response['choices'][0]['message']['content']
        
        # Extract the JSON part of the response
        sub_workflows_json = extract_json(response_content)
        
        if sub_workflows_json and isinstance(sub_workflows_json, list):
            return sub_workflows_json
        else:
            print("Error: LLM did not return a valid list of sub-workflows.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"An error occurred during LLM-based sub-workflow extraction: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sub-workflows from a workflow JSON file using an LLM.')
    parser.add_argument('--file_path', help='Path to the workflow JSON file.')
    parser.add_argument('--model', type=str, default='openrouter/google/gemma-3-27b-it:free', help='The model to use for extraction.')
    parser.add_argument('--temperature', type=float, default=0.0, help='The temperature for the LLM.')
    args = parser.parse_args()

    try:
        with open(args.file_path, 'r') as f:
            workflow_data = json.load(f)
        
        steps = workflow_data.get("steps", [])
        if not steps and isinstance(workflow_data, list):
            steps = workflow_data

        # --- Workflow Validation ---
        print("--- Validating Workflow ---")
        validation_failed = False
        
        duplicate_sids = find_duplicate_sids(steps)
        if duplicate_sids:
            print(f"Error: Found duplicate SIDs: {duplicate_sids}", file=sys.stderr)
            validation_failed = True

        broken_links = find_broken_links(steps)
        if broken_links:
            print(f"Error: Found broken links: {broken_links}", file=sys.stderr)
            validation_failed = True

        unreachable_steps = find_unreachable_steps(steps)
        if unreachable_steps:
            print(f"Warning: Found unreachable steps: {unreachable_steps}", file=sys.stderr)

        invalid_step_types = verify_step_types(steps)
        if invalid_step_types:
            print(f"Error: Found invalid step types: {invalid_step_types}", file=sys.stderr)
            validation_failed = True

        invalid_collections = verify_collection_values(steps)
        if invalid_collections:
            print(f"Error: Found invalid collection values: {invalid_collections}", file=sys.stderr)
            validation_failed = True
            
        missing_keys = detect_missing_keys(steps)
        if missing_keys:
            print(f"Error: Found steps with missing keys: {missing_keys}", file=sys.stderr)
            validation_failed = True

        if validation_failed:
            print("Workflow validation failed. Aborting.", file=sys.stderr)
            sys.exit(1)
        else:
            print("Workflow validation successful.")

        # --- Sub-workflow Extraction ---
        print("\n--- Extracting Sub-workflows ---")
        router = initialize_router()
        sub_workflows = extract_subflows_llm(steps, args, router)

        if sub_workflows:
            print(json.dumps(sub_workflows, indent=2))
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
