#!/bin/sh
# Evaluate and save sample input for synthesis
python train.py --model ai85pizzanotpizzanet64 --dataset pizza-not-pizza64 --data data/pizza_not_pizza64 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-pizza-qat8-q.pth.tar -8 --save-sample 10 --device MAX78000 "$@"