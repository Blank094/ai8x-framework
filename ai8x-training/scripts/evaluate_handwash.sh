#!/bin/sh
# Evaluate and save sample input for synthesis
python train_sklearn.py --model ai85handwashnet64 --dataset handwash64 --data data/handwash64 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-handwash-qat8-q.pth.tar -8 --save-sample 10 --device MAX78000 "$@"