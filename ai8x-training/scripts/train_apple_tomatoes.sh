#!/bin/sh

python train.py \
    --epochs 200 \
    --optimizer Adam \
    --lr 0.001 \
    --wd 0 \
    --deterministic \
    --compress policies/schedule-apple-tomatoes.yaml \
    --qat-policy policies/qat_policy_apple-tomatoes.yaml \
    --model ai85appletomatoesnet64 \
    --dataset apple-tomatoes64 \
    --data data/apple-tomatoes64 \
    --confusion \
    --param-hist \
    --embedding \
    --enable-tensorboard \
    --device MAX78000 \
    --batch-size 512 \
    "$@"