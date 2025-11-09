#!/bin/sh

python train.py \
    --epochs 300 \
    --optimizer Adam \
    --lr 0.001 \
    --wd 0 \
    --deterministic \
    --compress policies/schedule-apple-tomatoes.yaml \
    --qat-policy policies/qat_policy_apple-tomatoes.yaml \
    --model ai85pizzanotpizzanet64 \
    --dataset pizza-not-pizza64 \
    --data data/pizza_not_pizza64 \
    --confusion \
    --param-hist \
    --embedding \
    --enable-tensorboard \
    --device MAX78000 \
    --batch-size 512 \
    "$@"