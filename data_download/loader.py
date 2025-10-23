#!/usr/bin/env python3
"""
HuggingFace dataset download tool
"""

from datasets import load_dataset
import json
import sys
import os


def download_dataset(
    dataset_name: str,
    columns: str,
    template: str,
    max_samples: int,
    config: str = None,
    split: str = "train",
    output_dir: str = "data"
):
    """
    Download a HuggingFace dataset
    
    Args:
        dataset_name: Dataset name, e.g. "GAIR/lima"
        columns: Column names separated by commas, e.g. "question,answer"
        template: Merge template, e.g. "Question: {question}\nAnswer: {answer}"
        max_samples: Number of samples to download
        config: Dataset config name, e.g. "main" (optional)
        split: Dataset split, default "train"
        output_dir: Output directory, default "data"
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse column names
    columns_list = [col.strip() for col in columns.split(",")]
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if config:
        print(f"Using config: {config}")
        dataset = load_dataset(dataset_name, config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    print(f"Total samples: {len(dataset)}")
    print(f"Available columns: {dataset.column_names}")
    
    # Limit sample count
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Limited to: {max_samples} samples")
    
    print(f"Using columns: {columns_list}")
    print(f"Merge template: {template}")
    
    # Convert to prompt format
    result = []
    for item in dataset:
        template_vars = {}
        for col in columns_list:
            value = item[col]
            if isinstance(value, list):
                value = str(value)
            template_vars[col] = value
        
        prompt = template.format(**template_vars)
        result.append({"prompt": prompt})
    
    # Save as jsonl
    dataset_safe_name = dataset_name.replace("/", "_")
    output_file = os.path.join(output_dir, f"{dataset_safe_name}_{len(result)}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved to: {output_file}")
    print(f"Downloaded samples: {len(result)}")
    print(f"Sample example:")
    print(json.dumps(result[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python hf_downloader.py <dataset_name> <columns> <template> <max_samples> [config] [output_dir]")
        print('Example 1: python hf_downloader.py "GAIR/lima" "conversations" "{conversations}" 1000')
        print('Example 2: python hf_downloader.py "openai/gsm8k" "question,answer" "Q: {question}\\nA: {answer}" 2000 main')
        print('Example 3: python hf_downloader.py "GAIR/lima" "conversations" "{conversations}" 1000 "" data')
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    columns = sys.argv[2]
    template = sys.argv[3]
    max_samples = int(sys.argv[4])
    config = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    output_dir = sys.argv[6] if len(sys.argv) > 6 else "data"
    
    download_dataset(dataset_name, columns, template, max_samples, config, output_dir=output_dir)
