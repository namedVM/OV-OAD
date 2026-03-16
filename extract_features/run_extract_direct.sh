#!/bin/bash

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
ANNO_PATH=""
MODEL="ViT-B/16"
FPS=4.0

# ./extract_features/run_extract_direct.sh --input_dir data/Thumos14/videos --output_dir data/Thumos14D --anno_path data/Thumos14/annotation.jsonl
# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
        ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
        ;;
        --anno_path)
            ANNO_PATH="$2"
            shift 2
        ;;
        --model)
            MODEL="$2"
            shift 2
        ;;
        --fps)
            FPS="$2"
            shift 2
        ;;
        *)
            echo "Unknown argument: $1"
            exit 1
        ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --input_dir <path> --output_dir <path> [--anno_path <path>] [--model ViT-B/16] [--fps 4.0]"
    exit 1
fi

python extract_features/extract_feat_direct.py \
--input_dir "$INPUT_DIR" \
--output_dir "$OUTPUT_DIR" \
--anno_path "$ANNO_PATH" \
--model "$MODEL" \
--fps "$FPS"
