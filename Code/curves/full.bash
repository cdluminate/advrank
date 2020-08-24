#!/bin/bash
set -ex
OPTIRUN=optirun
which $OPTIRUN || OPTIRUN=''

#export CUDA_VISIBLE_DEVICES=7
#export RUTHLESS=10

ATTACKS=( ES:PGD-UT
          C+:PGD-W1
          C-:PGD-W1
          SPQ+:PGD-M1
          SPQ-:PGD-M1
	  )
models=(
		mnC_c2f2_trip:trained.vanilla/mnC_c2f2_trip.sdth
		mnC_c2f2_trip:trained.mintmaxe/mnC_c2f2_trip.sdth
	)


es=$(python3 -c 'for i in range(30+1): print(i/100)')

column(){
	for e in $es; do
		$OPTIRUN ./Attack.py -M $1 -A $2 -e $e
	done
}

for A in ${ATTACKS[@]}; do
	column ${models[0]} $A | tee mnC_c2f2_trip-vanilla.full.$A.log
done

for A in ${ATTACKS[@]}; do
	column ${models[1]} $A | tee mnC_c2f2_trip-defensive.full.$A.log
done
