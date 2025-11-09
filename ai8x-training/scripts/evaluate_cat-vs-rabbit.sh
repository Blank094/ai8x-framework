#!/bin/sh
# Evaluate and save sample input for synthesis
python train.py --model ai85catrabbitnet64 --dataset cat-rabbit64 --data data/cat-vs-rabbit64 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cat-vs-rabbit-qat8-q.pth.tar -8 --save-sample 10 --device MAX78000 "$@"            