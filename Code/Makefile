pep:
	fdfind '.*\.py$$' --exec autopep8 -avi

train-mnist:
	bash pipeline/train-mnist.sh

train-fashion:
	bash pipeline/train-fashion.sh

train-sop:
	bash pipeline/train-sop.sh

attack-mnist:
	bash pipeline/attack-mnist.sh
	rg Summary mn*.log

attack-fashion:
	bash pipeline/attack-fashion.sh
	rg Summary fa*.log

attack-sop:
	bash pipeline/attack-sop.sh
	rg Summary sop*.log

train-mnist-defense:
	echo === Warning: the vanila model under ./trained will be overwritten
	bash pipeline/train-mnist-defense.sh

train-fashion-defense:
	echo === Warning: the vanila model under ./trained will be overwritten
	bash pipeline/train-fashion-defense.sh

train-sop-defense:
	echo === Warning: the vanila model under ./trained will be overwritten
	bash pipeline/train-sop-defense.sh
