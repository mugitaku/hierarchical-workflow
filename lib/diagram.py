import json
import graphviz
import os
import sys

def generate_dot_from_json(data_input):
    """
    Generates a graphviz.Digraph object from a JSON string or a text file containing JSON.
    It robustly finds the JSON block even if the file has a text header.
    """
    data = None
    
    # 1. Try to parse the whole string as JSON for clean JSON files.
    try:
        data = json.loads(data_input)
    except json.JSONDecodeError:
        # If it fails, assume it's a text file with a header.
        # Find the start of the JSON block (first '{' or '[').
        first_brace = data_input.find('{')
        first_bracket = data_input.find('[')

        start_index = -1

        # Find the earliest occurrence of either character
        if first_brace != -1 and first_bracket != -1:
            start_index = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_index = first_brace
        elif first_bracket != -1:
            start_index = first_bracket
        
        # If a starting character was found, try to parse from there
        if start_index != -1:
            json_str = data_input[start_index:]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = None # It wasn't valid JSON after all

    # If JSON could not be parsed, return None
    if data is None:
        return None 

    # --- Below is the graph generation logic (unchanged) ---
    dot = graphviz.Digraph(comment='ScienceWorld Workflow', format='svg')
    dot.attr(rankdir='TB')  # Top to Bottom layout

    def process_steps(steps, graph):
        for step in steps:
            sid = step['sid']
            stype = step['step_type']
            
            if stype == 'action':
                label = f"{sid}\n[{step['action']}]"
                graph.node(sid, label, shape='box', style='filled', fillcolor='lightblue')
                if step.get('next_sid'):
                    graph.edge(sid, step['next_sid'])
            elif stype == 'branch':
                label = f"{sid}\n? {step['condition']} ?"
                graph.node(sid, label, shape='diamond', style='filled', fillcolor='lightyellow')
                if step.get('next_sid_if_true'):
                    graph.edge(sid, step['next_sid_if_true'], label="True")
                if step.get('next_sid_if_false'):
                    graph.edge(sid, step['next_sid_if_false'], label="False")
            elif stype == 'break':
                graph.node(sid, "BREAK", shape='circle', style='filled', fillcolor='red', fontcolor='white')
            elif stype == 'for_loop':
                with graph.subgraph(name=f'cluster_{sid}') as c:
                    c.attr(label=f"Loop: {step['collection']}\n(Iterator: {step['iterator']})", color='blue')
                    if 'steps' in step:
                        process_steps(step['steps'], c)
                        if len(step['steps']) > 0:
                            first_inner_step = step['steps'][0]['sid']
                            graph.edge(sid, first_inner_step, style='dashed', label="Start Loop")
                if step.get('next_sid'):
                    graph.node(sid, f"Start Loop: {sid}", shape='ellipse')
                    graph.edge(sid, step['next_sid'], label="Loop Done")

    if 'steps' in data:
        process_steps(data['steps'], dot)
    
    return dot

def render_dot(dot, output_filename):
    """
    Renders a graphviz.Digraph object to a file.
    """
    try:
        output_path = dot.render(output_filename, view=False, cleanup=True)
        print(f"Image generation complete: {output_path}")
    except Exception as e:
        print(f"Graphviz Error: {e}")
        print("Hint: Have you run 'sudo apt install graphviz'?")

def render_dot_source(dot_source, output_filename):
    """
    Renders a DOT source string to a file.
    """
    try:
        source = graphviz.Source(dot_source, format='svg')
        output_path = source.render(output_filename, view=False, cleanup=True)
        print(f"Image generation complete: {output_path}")
    except Exception as e:
        print(f"Graphviz Error: {e}")
        print("Hint: Have you run 'sudo apt install graphviz'?")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagram.py <workflow_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    print("Generating diagram...")

    # Try to generate dot from JSON (or text containing JSON)
    dot_from_json = generate_dot_from_json(file_content)

    output_filename_base = os.path.splitext(file_path)[0]

    if dot_from_json:
        # It was successfully parsed as JSON structure
        render_dot(dot_from_json, output_filename_base)
    else:
        # Assume it's a raw DOT file if JSON parsing completely failed
        render_dot_source(file_content, output_filename_base)
