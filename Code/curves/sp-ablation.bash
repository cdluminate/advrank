#!/bin/bash
set -x

sps=( 0 1 100 10000 )
models=( 
		mnC_c2f2_siamese
		mnC_c2f2_trip
		mnE_c2f2_siamese
		mnE_c2f2_trip
	)
attacks=(
	  SPQ+:PGD-M1 SPQ+:PGD-M2 SPQ+:PGD-M5 SPQ+:PGD-M10
	  SPQ-:PGD-M1 SPQ-:PGD-M2 SPQ-:PGD-M5 SPQ-:PGD-M10
  )


block_ () {
	model="$1"
	for A in ${attacks[@]}; do
		for sp in ${sps[@]}; do
			echo SP="$sp" ./Attack.py -M $model -A $A -e 0.3
			SP="$sp" ./Attack.py -M $model -A $A -e 0.3
		done
	done
}

for model in ${models[@]}; do
	block_ $model:trained.vanilla/$model.sdth | tee $model-vanilla-spablation.log
done
