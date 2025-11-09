#!/bin/sh
# Evaluate and save sample input for synthesis
python train.py --model ai85appletomatoesnet64 --dataset apple-tomatoes64 --data data/apple-tomatoes64 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-apple-tomatoes-qat8-q.pth.tar -8 --save-sample 10 --device MAX78000 "$@"