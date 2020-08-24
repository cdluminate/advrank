#!/bin/bash
set -x
#export CUDA_VISIBLE_DEVICES=7
#export RUTHLESS=10

ATTACKS=( C+:PGD-W1 C+:PGD-W2 C+:PGD-W5 C+:PGD-W10
          C-:PGD-W1 C-:PGD-W2 C-:PGD-W5 C-:PGD-W10
          Q+:PGD-M1 Q+:PGD-M2 Q+:PGD-M5 Q+:PGD-M10
          Q-:PGD-M1 Q-:PGD-M2 Q-:PGD-M5 Q-:PGD-M10
	  )

distance(){
	for A in ${ATTACKS[@]}; do
		DISTANCE=0 SP=0 python3 Attack.py -M $1 -v -e 0.3 -A $A
		DISTANCE=1 SP=0 python3 Attack.py -M $1 -v -e 0.3 -A $A
	done
}

model=$1
distance $model | tee ${model}-distance.log
