#!/bin/sh
# Parameters for code generation
DEVICE="MAX78000"
TARGET="C:\MaximSDK\Examples\MAX78000\CNN\new-apple-tomatoes\apple-tomatoes-demo-64x64"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Define the three required input files
QUANTIZED_MODEL="trained/ai85-apple-tomatoes-qat8-q.pth.tar"
YAML="networks/apple-tomatoes-hwc.yaml"
SAMPLE="tests/sample_apple-tomatoes64.npy"

# Generate embedded C code for MAX78000FTHR with optimizations
python ai8xize.py \
    --test-dir $TARGET \
    --prefix apple-tomatoes \
    --overwrite \
    --checkpoint-file $QUANTIZED_MODEL \
    --config-file $YAML \
    --sample-input $SAMPLE \
    --fifo \
    --softmax \
    $COMMON_ARGS \
    "$@"