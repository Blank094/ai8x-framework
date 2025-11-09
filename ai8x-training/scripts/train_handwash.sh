#!/bin/sh

python train.py \
    --epochs 50 \
    --optimizer Adam \
    --lr 0.001 \
    --wd 0 \
    --deterministic \
    --compress policies/schedule-handwash.yaml \
    --qat-policy policies/qat_policy_handwash.yaml \
    --model ai85handwashnet64 \
    --dataset handwash64 \
    --data data/handwash64 \
    --confusion \
    --param-hist \
    --embedding \
    --enable-tensorboard \
    --device MAX78000 \
    --batch-size 512 \
    "$@"