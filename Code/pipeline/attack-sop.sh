#!/bin/bash -e
ATTACKS=( C+:PGD-W1 C+:PGD-W2 C+:PGD-W5 C+:PGD-W10
          C-:PGD-W1 C-:PGD-W2 C-:PGD-W5 C-:PGD-W10
          SPQ+:PGD-M1 SPQ+:PGD-M2 SPQ+:PGD-M5 SPQ+:PGD-M10
          SPQ-:PGD-M1 SPQ-:PGD-M2 SPQ-:PGD-M5 SPQ-:PGD-M10
)
export RUTHLESS=10000

row(){
	# $1 -> epsilon
	for A in ${ATTACKS[@]}; do
		nohup python3 Attack.py -D cuda -M sopE_res18_trip -A ${A} -e $1 -v | tee sopE_res18_trip.${A}.${1}.log
	done
}

CUDA_VISIBLE_DEVICES=4 row 0.01 &
CUDA_VISIBLE_DEVICES=5 row 0.03 &
CUDA_VISIBLE_DEVICES=6 row 0.06 &
