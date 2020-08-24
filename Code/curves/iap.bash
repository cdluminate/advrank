#!/bin/bash
set -x

ATKS=(
	C+:PGD-W1
	C-:PGD-W1
	Q+:PGD-M1
	Q-:PGD-M1
)

model=$1

export IAP=1

_iap(){
for a in ${ATKS[@]}; do
	python3 Attack.py -M $1 -v -e 0.3 -A $a
done
}

_iap $model | tee $model-iap.log
