#!/bin/bash

# outdirectory configuration
OUTPUT_DIR="../data"
mkdir -p "$OUTPUT_DIR"

#================general huggingface======================

DOWNLOADER="../data_download/loader.py"
MAX_SAMPLES=2000   # How many samples(maximun) to download per dataset

# GAIR/lima
DATASET="GAIR/lima"
COLUMNS="conversations"
TEMPLATE="{conversations}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

# GSM8k 
DATASET="openai/gsm8k"
COLUMNS="question,answer"
TEMPLATE="{question}\n {answer}"
CONFIG="main"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "$CONFIG" "$OUTPUT_DIR"

#  MBPP 
DATASET="mbpp"
COLUMNS="text,code"
TEMPLATE="{text}\n {code}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="tatsu-lab/alpaca"
COLUMNS="instruction,input"
TEMPLATE="{instruction}\n {input}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="databricks/databricks-dolly-15k"
COLUMNS="instruction,context"
TEMPLATE="{instruction}\n {context}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="OpenAssistant/oasst1"
COLUMNS="text"
TEMPLATE="{text}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="WizardLMTeam/WizardLM_evol_instruct_70k"
COLUMNS="instruction"
TEMPLATE="{instruction}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="ai4privacy/pii-masking-200k"
COLUMNS="source_text"
TEMPLATE="{source_text}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="lmsys/toxic-chat"
COLUMNS="user_input"
TEMPLATE="{user_input}"
CONFIG="toxicchat0124"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "$CONFIG" "$OUTPUT_DIR"

DATASET="ScaleAI/fortress_public"
COLUMNS="adversarial_prompt"
TEMPLATE="{adversarial_prompt}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="code-rag-bench/ds1000"
COLUMNS="prompt"
TEMPLATE="{prompt}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"

DATASET="LLM-LAT/harmful-dataset"
COLUMNS="prompt"
TEMPLATE="{prompt}"
python3 "$DOWNLOADER" "$DATASET" "$COLUMNS" "$TEMPLATE" "$MAX_SAMPLES" "" "$OUTPUT_DIR"


# ================Routerbench=====================
DOWNLOADER="../data_download/routerbench_loader.py"
CONFIG="0shot"
python3 "$DOWNLOADER" "$CONFIG" "$MAX_SAMPLES" "$OUTPUT_DIR"

# ================Openchat=====================
DOWNLOADER="../data_download/openchat_loader.py"
FILENAME="openchat.train.text.json"
python3 "$DOWNLOADER" "$FILENAME" "$MAX_SAMPLES" "$OUTPUT_DIR"

# ================Unnatural=====================
DOWNLOADER="../data_download/unnatural_loader.py"
python3 "$DOWNLOADER" "$MAX_SAMPLES" "$OUTPUT_DIR"
