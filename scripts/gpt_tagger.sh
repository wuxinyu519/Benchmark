#!/bin/bash

#=================================================================

# OpenAI API Key
API_KEY="You-GPT-API-KEY-Here"

# Dir
INPUT_DIR="../data"
OUTPUT_DIR="${INPUT_DIR}/tagged_data"


# Model
MODEL="gpt-4o"
MAX_SAMPLES=2000 # How many samples(maximun) to process per file


TAGGER_SCRIPT="../gpt_infer/gpt_tagger.py"

# Check
[ -z "$API_KEY" ] && echo "Error: API_KEY not set" && exit 1
[ ! -d "$INPUT_DIR" ] && echo "Error: INPUT_DIR not found" && exit 1
[ ! -f "$TAGGER_SCRIPT" ] && echo "Error: TAGGER_SCRIPT not found" && exit 1

JSONL_COUNT=$(find "$INPUT_DIR" -name "*.jsonl" -type f | wc -l)
[ "$JSONL_COUNT" -eq 0 ] && echo "Error: No .jsonl files found" && exit 1

# Show info
echo "Found $JSONL_COUNT files in $INPUT_DIR:"
echo ""
find "$INPUT_DIR" -name "*.jsonl" -type f | while read file; do
    filename=$(basename "$file")
    line_count=$(wc -l < "$file")
    echo "  $filename ($line_count lines)"
done
echo ""
echo "Will process $MAX_SAMPLES samples per file using $MODEL"
echo ""

# Comfirm
read -p "Start inference? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# exec
python3 $TAGGER_SCRIPT $INPUT_DIR $API_KEY $MAX_SAMPLES $OUTPUT_DIR $MODEL




