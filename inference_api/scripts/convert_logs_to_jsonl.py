#!/usr/bin/env python3
"""
Convert JSON array log files to JSONL format (one JSON object per line)
"""
import json
import sys
from pathlib import Path

def convert_file(input_path, output_dir=None):
    """Convert a single JSON array file to JSONL format"""
    input_path = Path(input_path)
    
    # Set output path
    if output_dir is None:
        output_dir = input_path.parent
    output_path = Path(output_dir) / f"{input_path.stem}_jsonl{input_path.suffix}"
    
    print(f"Converting {input_path} to {output_path}")
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        try:
            # Read the entire file as JSON array
            log_entries = json.load(f_in)
            if not isinstance(log_entries, list):
                print(f"Warning: {input_path} does not contain a JSON array. Skipping.")
                return
                
            # Write each entry as a separate line
            for entry in log_entries:
                f_out.write(json.dumps(entry) + '\n')
                
            print(f"Successfully converted {len(log_entries)} entries")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing {input_path}: {e}")
            return

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file> [output_directory]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_file(input_path, output_dir)

if __name__ == "__main__":
    main()
