#!/bin/bash
set -ex
OPTIRUN=optirun
which $OPTIRUN || OPTIRUN=''

#export CUDA_VISIBLE_DEVICES=7
#export RUTHLESS=10

ATTACKS=( ES:PGD-UT
          C+:PGD-W1 C+:PGD-W2 C+:PGD-W5 C+:PGD-W10
          C-:PGD-W1 C-:PGD-W2 C-:PGD-W5 C-:PGD-W10
          SPQ+:PGD-M1 SPQ+:PGD-M2 SPQ+:PGD-M5 SPQ+:PGD-M10
          SPQ-:PGD-M1 SPQ-:PGD-M2 SPQ-:PGD-M5 SPQ-:PGD-M10
	  )
#          SPF:PGD-M2 SPF:PGD-M5 SPF:PGD-M10 )

fullrun(){
	local es=$(python3 -c 'for i in range(30+1): print(i/100)')
	for e in $es; do
		$OPTIRUN ./Attack.py -M $1 -A $2 -e $e
	done
}

typicalrun(){
	for e in 0.0 0.01 0.03 0.1 0.3; do
		$OPTIRUN ./Attack.py -M $1 -A $2 -e $e
	done
}

strong(){
	for A in ${ATTACKS[@]}; do
		$OPTIRUN ./Attack.py -M $1 -A $A -e 0.3 -v
	done
}

row(){
	for A in ${ATTACKS[@]}; do
		$OPTIRUN ./Attack.py -M $1 -A $A -e $2
	done
}

filter() {
	cat $1 | grep 'baseline='
}

transfer_ () {
	attack="$1"
	models=(
		mnC_lenet_trip:trained.vanilla/mnC_lenet_trip.sdth
		mnC_c2f2_siamese:trained.vanilla/mnC_c2f2_siamese.sdth
		mnC_c2f2_trip:trained.vanilla/mnC_c2f2_trip.sdth
		mnE_c2f2_trip:trained.vanilla/mnE_c2f2_trip.sdth
		mnC_res18_trip:trained.vanilla/mnC_res18_trip.sdth
		)
	for i in ${models[@]}; do
			for j in ${models[@]}; do
					./Attack.py -e 0.3 -A $attack -M $i -T $j
			done
	done
}

transfer() {
	attacks=(
		C+:PGD-W1
		C-:PGD-W1
		SPQ+:PGD-M1
		SPQ-:PGD-M1
		)
	for A in ${attacks[@]}; do
		transfer_ $A | tee mn-transfer-$A.log
	done
}

main() {
	local logpath=""
	case $1 in
		transfer)
			transfer
			;;
		strong)
			strong $2 2>&1 | tee > ${2}_all_strong_attack.log
			;;
		row)
			row $2 $3 2>&1 | tee > ${2}_all_${3}_attack.log
			;;
		block)
			model="$2"
			variant="$3"
			for e in 0.01 0.03 0.1 0.3; do
				if ! test -r ${model}-${e}-${variant}.log; then
					row $2:trained.${variant}/${2}.sdth ${e} 2>&1 \
						| tee > ${model}-${e}-${variant}.log
				fi
			done
			;;
		full)
			fullrun $2 $3 2>&1 | tee > ${1}_${2}_${3}.log
			;;
		typical)
			typicalrun $2 $3 2>&1 | tee > ${1}_${2}_${3}.log
			;;
		FULL)
			for ATK in ${ATTACKS[@]}; do
				logpath="full_${2}_${ATK}.log"
				if test -r $logpath || test -r $logpath.zst; then
					echo $logpath exists. skipping...
					continue;
				fi
				fullrun $2 $ATK 2>&1 | tee > $logpath
			done
			;;
		TYPICAL)
			for ATK in ${ATTACKS[@]}; do
				logpath="typical_${2}_${ATK}.log"
				if test -r $logpath || test -r $logpath.zst; then
					echo $logpath exists. skipping...
					continue;
				fi
				typicalrun $2 $ATK 2>&1 | tee > $logpath
			done
			;;
		filter)
			filter $2
			;;
		*)
			echo "???" && false
			;;
	esac
}

main $@
