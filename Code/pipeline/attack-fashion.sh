#!/bin/bash -e
ATTACKS=( C+:PGD-W1 C+:PGD-W2 C+:PGD-W5 C+:PGD-W10
          C-:PGD-W1 C-:PGD-W2 C-:PGD-W5 C-:PGD-W10
          SPQ+:PGD-M1 SPQ+:PGD-M2 SPQ+:PGD-M5 SPQ+:PGD-M10
          SPQ-:PGD-M1 SPQ-:PGD-M2 SPQ-:PGD-M5 SPQ-:PGD-M10
)

row(){
	# $1 -> epsilon
	for A in ${ATTACKS[@]}; do
		nohup python3 Attack.py -D cuda -M faC_c2f2_trip -A ${A} -e $1 -v | tee faC_c2f2_trip.${A}.${1}.log
	done
}

CUDA_VISIBLE_DEVICES=0 row 0.01 &
CUDA_VISIBLE_DEVICES=1 row 0.03 &
CUDA_VISIBLE_DEVICES=2 row 0.10 &
CUDA_VISIBLE_DEVICES=3 row 0.30 &
