#!/bin/sh

python train.py \
    --epochs 200 \
    --optimizer Adam \
    --lr 0.001 \
    --wd 0 \
    --deterministic \
    --compress policies/schedule-mask-hairnet.yaml \
    --qat-policy policies/qat_policy_mask_hairnet.yaml \
    --model ai85maskhairnetnet96 \
    --dataset mask_hairnet96 \
    --data data/mask_hairnet96 \
    --confusion \
    --param-hist \
    --embedding \
    --enable-tensorboard \
    --device MAX78000 \
    --batch-size 512 \
    --compiler-mode none \
    "$@" 