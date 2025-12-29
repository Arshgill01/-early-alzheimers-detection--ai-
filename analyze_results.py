
import json
import re

def analyze_notebook(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    print("--- EXTRACTED OUTPUTS ---")
    
    for i, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                # Handle stream output (stdout/stderr)
                if output.get('output_type') == 'stream':
                    text = "".join(output.get('text', []))
                    if any(x in text for x in ["Epoch", "Accuracy", "Loss", "classification report", "confusion matrix", "Test"]):
                         print(f"\n[Cell {i+1} Output]:\n{text}")
                
                # Handle execute_result (returned values)
                elif output.get('output_type') == 'execute_result':
                    data = output.get('data', {})
                    text = "".join(data.get('text/plain', []))
                    if any(x in text for x in ["Accuracy", "Loss", "Test"]):
                        print(f"\n[Cell {i+1} Result]:\n{text}")
                
                # Handle display_data (plots/images - just verify existence)
                elif output.get('output_type') == 'display_data':
                     print(f"\n[Cell {i+1} Display Data]: Contains image/plot")

if __name__ == "__main__":
    analyze_notebook('output.ipynb')
