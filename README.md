# Benchmark Dataset Collection & Tagging

A comprehensive toolkit for downloading datasets and generating ground truth tags using GPT models.

## Quick Start

### Prerequisites

```bash
create a new env
pip install -r requirements.txt
```

### Step 1: Download Datasets

```bash
cd scripts
bash data_loader.sh
```

**Main Parameters** (edit in `data_loader.sh`):
- `MAX_SAMPLES`: Number of samples per dataset (default: 2000)
- `OUTPUT_DIR`: Output directory (default: ../data)

### Step 2: Generate Tags with GPT

```bash
cd scripts
bash gpt_tagger.sh
```

**Main Parameters** (edit in `gpt_tagger.sh`):
- `API_KEY`: Your OpenAI API key (required)
- `MODEL`: GPT model to use (default: gpt-4o)
- `MAX_SAMPLES`: Samples to process per file (default: 2)
- `INPUT_DIR`: Input directory (default: ../data)
- `OUTPUT_DIR`: Output directory (default: tagged_data)

## Project Structure

```
benchmark/
├── scripts/
│   ├── data_loader.sh          # Download datasets scripts
│   └── gpt_tagger.sh           # Generate tags with GPT scripts
├── data_download/              # Dataset loader files
├── gpt_infer/                  # GPT tagging 
├── data/                       # Downloaded datasets (generated)
│   └── tagged_data/            # Tagged results (generated)

```              

## Features

- **Automatic Dataset Type Detection**: Detects dataset types (code, math, PII, toxic, etc.) and uses specialized prompts
- **Checkpoint Resumption**: Saves progress every 5 samples - automatically resumes if interrupted
- **Token Truncation**: For long texts (>600 tokens), keeps first 300 + last 300 tokens to reduce costs

## Output Format

### Downloaded Data (`data/*.jsonl`)
```json
{"prompt": "instruction text"}
```

### Tagged Data (`data/tagged_data/*_tagged.jsonl`)
```json
{
  "prompt": "instruction text",
  "ground_truth": [
    {"tag": "Domain", "explanation": "..."},
    {"tag": "Task Type", "explanation": "..."},
    {"tag": "Difficulty", "explanation": "..."},
    {"tag": "Language", "explanation": "..."},
    {"tag": "Topic 1", "explanation": "..."}
  ]
}
```

## Supported Datasets

| Category | Datasets |
|----------|----------|
| Code | mbpp, ds1000 |
| Math | gsm8k |
| General | alpaca, dolly, wizardlm, lima, openchat... |
| Security | fortress |
| Privacy | pii-masking |
| Toxic | toxic-chat |
| Company confidential | contract |
| harmful | LLM-LAT/harmful-dataset |

