import nbformat as nbf
import os
import re

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        code = f.read()

    nb = nbf.v4.new_notebook()
    
    # Simple partitioning based on markers or large comments
    # We want to keep imports together, then functions, then main.
    
    sections = re.split(r'(?m)^# %%.*', code)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if it's a docstring section
        if section.startswith('"""') and section.endswith('"""'):
            md = section.strip('"""').strip()
            nb.cells.append(nbf.v4.new_markdown_cell(md))
        elif section.startswith('=' * 10):
            # Header comments
            lines = section.split('\n')
            title = lines[0].strip('= ')
            content = '\n'.join(lines[1:])
            nb.cells.append(nbf.v4.new_markdown_cell(f"## {title}\n{content}"))
        else:
            nb.cells.append(nbf.v4.new_code_cell(section))

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    convert_py_to_ipynb('time_series_analysis.py', 'time_series_analysis.ipynb')
    print("Successfully converted time_series_analysis.py to time_series_analysis.ipynb")
