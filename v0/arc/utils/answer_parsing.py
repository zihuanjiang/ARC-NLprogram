# arc/utils/answer_parsing.py

import json
import re

def parse_ascii_grid(llm_output):
    # Try parsing as JSON first
    try:
        cleaned_output = llm_output.strip()
        # robust markdown extraction
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
        
        data = json.loads(cleaned_output)
        
        grid = None
        if isinstance(data, list):
            grid = data
        elif isinstance(data, dict):
            if "output_grid" in data:
                grid = data["output_grid"]
            elif "grid" in data:
                grid = data["grid"]
            elif "prediction" in data:
                grid = data["prediction"]
            elif "output" in data:
                grid = data["output"]
        
        if grid is not None:
            # Handle list of strings format ["000", "000"]
            if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], str):
                parsed_grid = []
                for row in grid:
                    parsed_grid.append([int(c) for c in row.strip()])
                return parsed_grid
            # Handle standard list of lists [[0,0], [0,0]]
            elif isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                return grid
            # Handle empty grid
            elif isinstance(grid, list) and len(grid) == 0:
                return []
            else:
                # Grid extracted but in unexpected format
                print(f"Warning: Grid in unexpected format (type: {type(grid)}), falling back to regex")
                
    except json.JSONDecodeError:
        pass

    # Fallback to regex for plain text grids
    match = re.search(r'((?:\d+\n*)+)', llm_output)
    if not match:
        print("Warning: Could not find a grid in the LLM output.")
        return []
    
    grid_str = match.group(1).strip()
    
    # Convert the string block into a list of lists of ints
    grid = []
    for line in grid_str.split('\n'):
        if line.strip():
            grid.append([int(char) for char in line.strip()])
            
    return grid
