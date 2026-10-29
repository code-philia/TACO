import json
from pathlib import Path

def initialize_knowledge_graph(output_dir):
    """
    Initialize the detail knowledge graph file from the base architecture graph.

    1. Check whether `knowledge_graph.json` exists.
    2. If it exists, clear `file_mappings` and save it as `knowledge_graph_detail.json`.
    3. Otherwise, remind the user to run the agent exploration phase first.
    """
    output_path = Path(output_dir)
    old_kb = output_path / "knowledge_graph.json"
    new_kb_detail = output_path / "knowledge_graph_detail.json"

    if old_kb.exists():
        print(f"🔍 [Init] Found existing KB at {old_kb}")
        with open(old_kb, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Preserve the architecture skeleton and clear file mappings.
        # Phase B (batch mode) can then regenerate file-level details without conflicts.
        initial_count = len(data.get('file_mappings', []))
        data['file_mappings'] = [] 
        
        with open(new_kb_detail, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ [Init] Successfully cleared {initial_count} mappings.")
        print(f"✅ [Init] New base for Batch Mode saved to: {new_kb_detail.name}")
        return True
    else:
        print(f"⚠️ [Init] No base knowledge_graph.json found in {output_path}.")
        print("   You may need to run 'agent' mode first to build the architecture.")
        return False

if __name__ == "__main__":
    # Local smoke-test entrypoint.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Path to repo2docAgent-output")
    args = parser.parse_args()
    initialize_knowledge_graph(args.output_dir)
