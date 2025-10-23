#!/usr/bin/env python3
"""
Download openchat_sharegpt4_dataset and extract human content
"""

import json
import sys
import os
from huggingface_hub import hf_hub_download


def download_openchat(filename, max_samples=None, output_dir="data"):
    """
    Download the openchat dataset file and extract human content
    
    Args:
        filename: File name, e.g. "openchat.train.text.json"
        max_samples: Maximum number of samples to download, None for all
        output_dir: Output directory, default "data"
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    repo_id = "openchat/openchat_sharegpt4_dataset"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}/{filename}")
    print(f"{'='*60}")
    
    try:
        # Download file from HuggingFace Hub
        print("Downloading file...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        print(f"File downloaded to: {local_path}")
        
        # Read JSON file
        print("Reading data...")
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Data loaded successfully")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Total samples: {len(data)}")
        else:
            print(f"Data structure: {type(data)}")
        
        # Limit number of items
        items = data if isinstance(data, list) else [data]
        
        # Inspect first sample structure
        if items:
            print(f"\nStructure of first sample:")
            first_item = items[0]
            if isinstance(first_item, dict):
                print(f"Keys: {first_item.keys()}")
                print(f"Example data:")
                print(json.dumps(first_item, ensure_ascii=False, indent=2)[:500])
            else:
                print(f"Type: {type(first_item)}")
                print(str(first_item)[:500])
        
        if max_samples and max_samples < len(items):
            items = items[:max_samples]
            print(f"\nLimited to: {max_samples} samples")
        
        # Extract human content
        result = []
        for idx, item in enumerate(items):
            human_content = None
            
            # If item is a string
            if isinstance(item, str):
                text = item.strip()
                
                # Method 1: match "Human:" or "<s>Human:"
                if 'Human:' in text:
                    parts = text.split('Human:', 1)
                    if len(parts) > 1:
                        human_content = parts[1].strip()
                        # Stop at assistant-related markers
                        for separator in ['Assistant:', 'GPT:', 'AI:', '<|im_start|>assistant']:
                            if separator in human_content:
                                human_content = human_content.split(separator)[0].strip()
                                break
                
                # Method 2: use entire text if no "Human:" found
                if not human_content:
                    human_content = text
            
            # If item is a dictionary
            elif isinstance(item, dict):
                if 'human' in item:
                    human_content = item['human']
                elif 'conversations' in item:
                    convs = item['conversations']
                    if isinstance(convs, list):
                        for msg in convs:
                            if isinstance(msg, dict):
                                if msg.get('from') == 'human' or msg.get('role') == 'user':
                                    human_content = msg.get('value') or msg.get('content')
                                    break
            
            # Add to result
            if human_content:
                result.append({"prompt": human_content})
        
        print(f"Extracted {len(result)} human entries")
        
        # Save as jsonl
        output_file = os.path.join(output_dir, f"openchat_{len(result)}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Show sample
        if result:
            print(f"\nSample:")
            sample = result[0]['prompt']
            print(sample[:300] + "..." if len(sample) > 300 else sample)
        
    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python openchat_loader.py <filename> <max_samples> [output_dir]")
        print('Example: python openchat_loader.py openchat.train.text.json 2000 data')
        print('\nAvailable files:')
        print('  - openchat.train.text.json')
        print('  - openchat.eval.text.json')
        print('  - openchat_8192.train.text.json')
        sys.exit(1)
    
    filename = sys.argv[1]
    max_samples = int(sys.argv[2])
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    
    download_openchat(filename, max_samples, output_dir)
