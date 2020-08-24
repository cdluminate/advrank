#!/bin/bash
set -ex

#export CUDA_VISIBLE_DEVICES=7
#export RUTHLESS=10

ATTACKS=( SPF:PGD-M2 SPF:PGD-M5 SPF:PGD-M10 )
models=(
	mnC_c2f2_siamese
	mnC_c2f2_trip
	mnE_c2f2_siamese
	mnE_c2f2_trip
	)
ES=( 
#	0
	0.3 
)

block(){
	for E in ${ES[@]}; do
		for A in ${ATTACKS[@]}; do
			./Attack.py -M $1:trained.vanilla/$1.sdth -A $A -e $E -v
		done
	done
}

for M in ${models[@]}; do
	block $M | tee $M-SPF.log
done
