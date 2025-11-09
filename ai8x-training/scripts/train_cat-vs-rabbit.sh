#!/bin/sh

python train.py \
    --epochs 20 \
    --optimizer Adam \
    --lr 0.001 \
    --wd 0 \
    --deterministic \
    --compress policies/schedule-cat-vs-rabbit.yaml \
    --qat-policy policies/qat_policy_cat-vs-rabbit.yaml \
    --model ai85catrabbitnet64 \
    --dataset cat-rabbit64 \
    --data data/cat-vs-rabbit64 \
    --confusion \
    --param-hist \
    --embedding \
    --enable-tensorboard \
    --device MAX78000 \
    --batch-size 512 \
    "$@"