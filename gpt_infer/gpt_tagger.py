#!/usr/bin/env python3


import os
import json
import sys
from pathlib import Path
from openai import OpenAI
import time
import tiktoken


class DatasetTagger:
    
    def __init__(self, api_key: str, model: str = "gpt-4", output_dir: str = "tagged_data"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.output_dir = output_dir
        self.encoding = tiktoken.encoding_for_model(model)
        os.makedirs(output_dir, exist_ok=True)
    
    def truncate_text(self, text: str, max_tokens: int = 600) -> str:
        """
        keep first 300 and last 300 token
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        half = max_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
        
        return self.encoding.decode(truncated_tokens)
    
    def get_dataset_type(self, filename: str) -> str:
        """Determine template type based on filename keywords"""
        filename_lower = filename.lower()
        
        if any(k in filename_lower for k in ['mbpp', 'ds1000']):
            return "code"
        if any(k in filename_lower for k in ['gsm8k']):
            return "math"
        if any(k in filename_lower for k in ['pii']):
            return "pii"
        if any(k in filename_lower for k in ['toxic']):
            return "toxic"
        if any(k in filename_lower for k in ['contract']):
            return "confidential"
        if any(k in filename_lower for k in ['fortress']):
            return "security"
        if any(k in filename_lower for k in ['harmful']):
            return "harmful"
        
        return "general"
    
    def get_prompt_template(self, dataset_type: str) -> str:
        """prompt template"""

        base_instructions = """Generate semantic tags from multiple perspectives.
                Choose the perspectives dynamically from the instruction itself.
                Explain each tag briefly.
                Please return in English.
                Return only valid JSON in this exact format:
                [
                  {{"perspective": str, "tag": str, "explanation": str}}
                ]"""

        templates = {
        "code": """You are a tagging system that labels the key intent of a **coding-related** user instruction.
                {instruction}
                """ + base_instructions,

        "math": """You are a tagging system that labels the key intent of a **math-related** user instruction.
                {instruction}
                """ + base_instructions,

        "pii": """You are a tagging system that labels the key intent of a **PII (Personally Identifiable Information) related** user instruction.
                {instruction}

                If the instruction contains **personal-sensitive information**  
                **do NOT reveal or restate that information**.  
                """ + base_instructions,

        "toxic": """You are a tagging system that labels the key intent of a **toxic-related** user instruction.
                {instruction}

                If the instruction contains **toxic-sensitive information**  
                **do NOT reveal or restate that information**.  
                """ + base_instructions,

        "confidential": """You are a tagging system that labels the key intent of a **company confidential-related** user instruction.
                {instruction}

                If the instruction contains **company confidential-sensitive information**  
                **do NOT reveal or restate that information**.  
                """ + base_instructions,

        "security": """You are a tagging system that labels the key intent of a **national security-related** user instruction.
                {instruction}

                If the instruction contains **security-sensitive information**  
                **do NOT reveal or restate that information**.  
                """ + base_instructions,

        "harmful": """You are a tagging system that labels the key intent of a **national security-related** user instruction.
                {instruction}

                If the instruction contains **personal, toxic, harmful, company confidential, or security-sensitive information**  
                **do NOT reveal or restate that information**.  
                """ + base_instructions,

        "general": """You are a tagging system that labels the key intent of a user instruction.
                {instruction}
                """ + base_instructions
        }
        
        return templates.get(dataset_type, templates["general"])
    
    def generate_tags(self, instruction: str, dataset_type: str):
        """generate tags"""
        truncated_instruction = self.truncate_text(instruction)
        
        prompt = self.get_prompt_template(dataset_type).format(instruction=truncated_instruction)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful tagging assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            return json.loads(content)
            
        except Exception as e:
            print(f" Error: {e}")
            return []
    
    def process_file(self, filepath: str, max_samples: int = None):
        """Single file processing"""
        filename = os.path.basename(filepath)
        dataset_type = self.get_dataset_type(filename)
        
        print(f"\n{'='*60}")
        print(f"File: {filename} | Type: {dataset_type}")
        print(f"{'='*60}")
        
        # Read data
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
        
        print(f"Total #: {len(data)}")
        
        # Limit samples
        if max_samples and max_samples < len(data):
            data = data[:max_samples]
            print(f"Maximun processing: {max_samples}")
        
        
        output_path = os.path.join(
            self.output_dir, 
            filename.replace('.jsonl', '_tagged.jsonl')
        )
        
        # check already processed
        processed_count = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for _ in f)
            print(f"Processed: {processed_count} samples, Continue...")
        
        # generated batch
        batch = []
        for idx, item in enumerate(data[processed_count:], processed_count + 1):
            instruction = item.get('prompt', '')
            if not instruction:
                continue
            
            print(f"[{idx}/{len(data)}]", end=' ')
            
            tags = self.generate_tags(instruction, dataset_type)
            
            batch.append({
                "prompt": instruction,
                "ground_truth": tags
            })
            
            print(f"{len(tags)} tags")
            
            # save every 10 data
            if len(batch) >= 10:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for result in batch:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"  Already save {idx} records.")
                batch = []  
            
            time.sleep(0.5)
        
        # save remaining
        if batch:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in batch:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nSaved: {output_path}")
    
    def process_directory(self, input_dir: str, max_samples: int = None):
        """Directory processing"""
        jsonl_files = list(Path(input_dir).glob('*.jsonl'))
        
        print(f"\nFound {len(jsonl_files)} files")
        
        for filepath in jsonl_files:
            self.process_file(str(filepath), max_samples)
        
        print(f"\n{'='*60}")
        print(f"Done, output: {self.output_dir}")
        print(f"{'='*60}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python gpt_tagger.py <input_dir> <api_key> [max_samples] [output_dir] [model]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    api_key = sys.argv[2]
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "tagged_data"
    model = sys.argv[5] if len(sys.argv) > 5 else "gpt-4"
    
    tagger = DatasetTagger(api_key, model, output_dir)
    tagger.process_directory(input_dir, max_samples)


if __name__ == "__main__":
    main()
