import json

# Load the notebook file
notebook_file = "DEEP.ipynb"

try:
    # Open and read the notebook file
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)

    # Extract code cells
    code_cells = [
        cell['source']
        for cell in notebook_data['cells']
        if cell['cell_type'] == 'code'
    ]

    # Combine code cells into a single script
    extracted_code = "\n\n".join(["".join(cell) for cell in code_cells])

    # Save the extracted code to a Python file
    output_file = "extracted_code.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_code)

    print(f"Code cells have been extracted and saved to {output_file}.")
except Exception as e:
    print(f"An error occurred: {e}")
