#!/bin/sh
# Evaluate and save sample input for synthesis
python train.py --model ai85maskhairnetnet128 --dataset mask_hairnet128 --data data/mask_hairnet128 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-mask-hairnet-qat8-q.pth.tar -8 --save-sample 10 --device MAX78000 "$@" 