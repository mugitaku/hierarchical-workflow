import argparse
import os
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Workflow Generator & Refiner")
    parser.add_argument("--model", type=str, help="Specify the model_name in litellm.yaml")
    parser.add_argument("--user_env_file", type=str, help="Text file describing the agent's environment")
    parser.add_argument("--user_task_file", type=str, help="Text file describing the agent's task")
    parser.add_argument("--sys_main_file", type=str, default="prompts/sys/main.txt", help="System Prompt for main workflow")
    parser.add_argument("--sys_sub_file", type=str, default="prompts/sys/sub.txt", help="System Prompt for sub workflow")
    parser.add_argument("--format_file", type=str, default="prompts/sys/format.txt", help="System Prompt for format")
    parser.add_argument("--example_file", type=str, default="prompts/sys/example.txt", help="System Prompt for few shot example")
    parser.add_argument("--actions_file", type=str, default="prompts/known_actions-simple.json", help="Action definition file path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth of recursive decomposition")
    parser.add_argument("--disable_example", action='store_true', help="Flag to disable few-shot example")
    parser.add_argument("--disable_db", action='store_true', help="Flag to disable DB search")
    parser.add_argument("--disable_subflow", action='store_true', help="Flag to disable subflow generation")
    parser.add_argument("--generate_diagram", action='store_true', help="Execute diagram.py to generate a workflow diagram upon completion")
    return parser.parse_args()

def load_file_content(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_and_format_actions(filepath, action_limited=False, action_types=None):
    if not os.path.exists(filepath):
        print(f"Warning: Action list file not found: {filepath}")
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            actions_data = json.load(f)
        
        if action_types:
            actions_data = [item for item in actions_data if item.get('type') in action_types]

        header = "You can use only the following actions:\n" if action_limited else "You can use the following and other actions:\n"
        actions_str = "".join([f"* {item['example']}: {item['description']}\n" for item in actions_data])
        return header + actions_str
    except Exception as e:
        print(f"Error loading action list: {e}")
        return ""

def get_action_definitions(filepath):
    """Get the entire actions_list.json as a list of dictionaries"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in action definition file: {e}")
        return []
