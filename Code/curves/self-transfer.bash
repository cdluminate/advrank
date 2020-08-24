#!/bin/bash
set -ex

transfer () {
	attack="$1"
	models=(
		mnC_c2f2_trip:trained.vanilla/mnC_c2f2_trip.sdth
		mnC_c2f2_trip:trained.vanilla/mnC_c2f2_trip-2.sdth
		mnC_c2f2_trip:trained.vanilla/mnC_c2f2_trip-3.sdth
		mnC_c2f2_trip:trained.mintmaxe/mnC_c2f2_trip.sdth
		mnC_c2f2_trip:trained.mintmaxe/mnC_c2f2_trip-2.sdth
		)
	for i in ${models[@]}; do
			for j in ${models[@]}; do
					./Attack.py -e 0.3 -A $attack -M $i -T $j
			done
	done
}

attacks=(
	C+:PGD-W1
	C-:PGD-W1
	SPQ+:PGD-M1
	SPQ-:PGD-M1
	)
for A in ${attacks[@]}; do
	transfer $A | tee self-transfer-$A.log
done
