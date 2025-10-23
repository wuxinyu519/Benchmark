#!/usr/bin/env python3
"""
Download the unnatural-instructions-full dataset and extract instruction_with_input
"""

from datasets import load_dataset
import json
import sys
import os


def download_unnatural(max_samples=None, output_dir="data"):
    """
    Download unnatural-instructions-full and extract instruction_with_input
    
    Args:
        max_samples: Maximum number of samples to download, None for all
        output_dir: Output directory, default "data"
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = "mrm8488/unnatural-instructions-full"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split="train")
        
        print("Dataset loaded successfully")
        print(f"Total samples: {len(dataset)}")
        print(f"Available columns: {dataset.column_names}")
        
        # Inspect the first item structure
        if len(dataset) > 0:
            print(f"\nFirst item structure:")
            first_item = dataset[0]
            print(f"Keys: {first_item.keys()}")
            if 'instances' in first_item:
                print(f"Instances type: {type(first_item['instances'])}")
                if first_item['instances']:
                    print(f"First instance: {first_item['instances'][0]}")
        
        # Limit the number of samples
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"\nLimited to: {max_samples} samples")
        
        # Extract instruction_with_input
        result = []
        total_instances = 0
        
        for item in dataset:
            instances = item.get('instances', [])
            if isinstance(instances, list):
                for instance in instances:
                    if isinstance(instance, dict):
                        instruction_with_input = instance.get('instruction_with_input')
                        if instruction_with_input:
                            result.append({"prompt": instruction_with_input})
                            total_instances += 1
        
        print(f"\nExtracted {len(result)} instruction_with_input samples")
        print(f"From {len(dataset)} tasks")
        
        # Save as jsonl
        output_file = os.path.join(output_dir, f"unnatural_{len(result)}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Display one example
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
    if len(sys.argv) < 2:
        print("Usage: python unnatural_loader.py <max_samples> [output_dir]")
        print('Example: python unnatural_loader.py 2000 data')
        print('Note: max_samples refers to the number of tasks, each task may have multiple instances')
        sys.exit(1)
    
    max_samples = int(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    download_unnatural(max_samples, output_dir)
