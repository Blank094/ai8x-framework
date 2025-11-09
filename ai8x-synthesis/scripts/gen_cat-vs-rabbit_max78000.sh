#!/bin/sh
# Parameters for code generation
DEVICE="MAX78000"
TARGET="C:\MaximSDK\Examples\MAX78000\CNN\new-cat-vs-rabbit\cat-vs-rabbit-demo-64x64"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Define the three required input files
QUANTIZED_MODEL="trained/ai85-cat-vs-rabbit-qat8-q.pth.tar"
YAML="networks/cat-vs-rabbit-hwc.yaml"
SAMPLE="tests/sample_cat-rabbit64.npy"

# Generate embedded C code for MAX78000FTHR with optimizations
python ai8xize.py \
    --test-dir $TARGET \
    --prefix cat-vs-rabbit \
    --overwrite \
    --checkpoint-file $QUANTIZED_MODEL \
    --config-file $YAML \
    --sample-input $SAMPLE \
    --fifo \
    --softmax \
    $COMMON_ARGS \
    "$@"