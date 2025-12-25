import json
import graphviz
import os
import sys

def generate_dot_from_json(data_input):
    """
    Generates a graphviz.Digraph object from a JSON string or a text file containing JSON.
    """
    data = None
    
    # 1. そのままJSONとしてパースを試みる (互換性維持)
    try:
        data = json.loads(data_input)
    except json.JSONDecodeError:
        # 2. テキスト形式 ("Final Workflow:" ヘッダーが含まれる場合) からの抽出を試みる
        target_header = "Final Workflow:"
        if target_header in data_input:
            # ヘッダー以降のテキストを取得
            header_index = data_input.find(target_header)
            content_after_header = data_input[header_index:]
            
            # 最初の '{' を探す
            json_start_index = content_after_header.find('{')
            if json_start_index != -1:
                json_str = content_after_header[json_start_index:]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

    # JSONが見つからなかった、またはパースできなかった場合
    if data is None:
        return None 

    # --- 以下、グラフ生成ロジック (変更なし) ---
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
        print(f"画像生成完了: {output_path}")
    except Exception as e:
        print(f"Graphviz Error: {e}")
        print("ヒント: 'sudo apt install graphviz' は実行しましたか？")

def render_dot_source(dot_source, output_filename):
    """
    Renders a DOT source string to a file.
    """
    try:
        source = graphviz.Source(dot_source, format='svg')
        output_path = source.render(output_filename, view=False, cleanup=True)
        print(f"画像生成完了: {output_path}")
    except Exception as e:
        print(f"Graphviz Error: {e}")
        print("ヒント: 'sudo apt install graphviz' は実行しましたか？")


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