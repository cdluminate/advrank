#!/bin/sh -e
tmux new-session -s "train-mnist-defense" "gpustat -i1"\; split-window "htop"\;\
	new-window -n "CC" "CUDA_VISIBLE_DEVICES=0 python3 Train.py -D cuda -M mnC_c2f2_siamese -A 0.3; sh"\;\
	new-window -n "CT" "CUDA_VISIBLE_DEVICES=1 python3 Train.py -D cuda -M mnC_c2f2_trip    -A 0.3; sh"\;\
	new-window -n "EC" "CUDA_VISIBLE_DEVICES=2 python3 Train.py -D cuda -M mnE_c2f2_siamese -A 0.3; sh"\;\
	new-window -n "ET" "CUDA_VISIBLE_DEVICES=3 python3 Train.py -D cuda -M mnE_c2f2_trip    -A 0.3; sh"\;\
	detach
