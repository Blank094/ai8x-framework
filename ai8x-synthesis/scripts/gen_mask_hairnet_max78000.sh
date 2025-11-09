#!/bin/sh
# Parameters for code generation
DEVICE="MAX78000"
TARGET="C:\MaximSDK\Examples\MAX78000\CNN\new3_mask_hairnet\mask-hairnet-128x128"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Define the three required input files
QUANTIZED_MODEL="trained/ai85-mask-hairnet-qat8-q.pth.tar"
YAML="networks/mask-hairnet-hwc.yaml"
SAMPLE="tests/sample_mask_hairnet128.npy"

# Generate embedded C code for MAX78000FTHR with optimizations
python ai8xize.py \
    --test-dir $TARGET \
    --prefix mask_hairnet \
    --overwrite \
    --checkpoint-file $QUANTIZED_MODEL \
    --config-file $YAML \
    --sample-input $SAMPLE \
    --fifo \
    --softmax \
    $COMMON_ARGS \
    "$@" 