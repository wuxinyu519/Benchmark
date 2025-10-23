#!/usr/bin/env python3
"""
Download .pkl files from the withmartian/routerbench dataset
"""

import pickle
import json
import sys
import os
from huggingface_hub import hf_hub_download


def download_routerbench_shots(shot_config, max_samples=None, output_dir="data"):
    """
    Download a specific shot configuration from the routerbench dataset
    
    Args:
        shot_config: Shot configuration name, e.g. "0shot", "5shot", "raw"
        max_samples: Maximum number of samples to download, None for all
        output_dir: Output directory, default "data"
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    repo_id = "withmartian/routerbench"
    
    # File name mapping
    filename = f"routerbench_{shot_config}.pkl"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id} - {filename}")
    print(f"{'='*60}")
    
    try:
        # Download file from HuggingFace Hub
        print("Downloading file...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        print(f"  File downloaded to: {local_path}")
        
        # Read pickle file
        print("Reading data...")
        with open(local_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  Data loaded successfully")
        print(f"Data type: {type(data)}")
        
        # Process data (adjust based on actual structure)
        if isinstance(data, list):
            print(f"Total samples: {len(data)}")
            items = data
        elif isinstance(data, dict):
            print(f"Available keys: {data.keys()}")
            # Try to locate the list of data
            if 'data' in data:
                items = data['data']
            elif 'examples' in data:
                items = data['examples']
            else:
                items = list(data.values())[0] if data else []
            print(f"Total samples: {len(items)}")
        else:
            # Handle DataFrame or similar structures
            try:
                import pandas as pd
                if hasattr(data, 'to_dict'):
                    df = data
                    print(f"Total samples: {len(df)}")
                    print(f"Available columns: {df.columns.tolist()}")
                    
                    # Extract only prompt column if available
                    if 'prompt' in df.columns:
                        items = df['prompt'].tolist()
                        print(f"  Extracted 'prompt' column")
                    else:
                        print(f"❌ 'prompt' column not found")
                        items = []
                else:
                    items = [data]
            except:
                items = [data]
        
        # Limit number of samples
        if max_samples and max_samples < len(items):
            items = items[:max_samples]
            print(f"Limited to: {max_samples} samples")
        
        # Convert to prompt format and save
        result = []
        for item in items:
            if isinstance(item, str):
                prompt_text = item
            elif isinstance(item, dict) and 'prompt' in item:
                prompt_text = item['prompt']
                if isinstance(prompt_text, list):
                    prompt_text = str(prompt_text)
            elif isinstance(item, list):
                prompt_text = str(item)
            elif isinstance(item, dict):
                prompt_text = json.dumps(item, ensure_ascii=False)
            else:
                prompt_text = str(item)
            
            result.append({"prompt": prompt_text})
        
        # Save as jsonl
        output_file = os.path.join(output_dir, f"routerbench_{shot_config}_{len(result)}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  Saved to: {output_file}")
        print(f"  Total samples: {len(result)}")
        
        # Show first sample
        if result:
            print(f"\nSample:")
            sample = result[0]['prompt']
            print(sample[:300] + "..." if len(sample) > 300 else sample)
        
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"\nAvailable files:")
        print("  - routerbench_0shot.pkl")
        print("  - routerbench_5shot.pkl")
        print("  - routerbench_raw.pkl")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python routerbench_loader.py <shot_config> <max_samples> [output_dir]")
        print('Example: python routerbench_loader.py 0shot 2000 data')
        print('Available configs: 0shot, 5shot, raw')
        sys.exit(1)
    
    shot_config = sys.argv[1]  # e.g. "0shot", "5shot", "raw"
    max_samples = int(sys.argv[2])
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    
    download_routerbench_shots(shot_config, max_samples, output_dir)
