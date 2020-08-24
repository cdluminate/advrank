/Summary/{
	if (c % 5 == 0) {
		print "";
	}
	if (c % 6 == 0) {
		printf "\\cellcolor{black!10}50$\\rightarrow$%.1f &", 100*$NF;
	} else {
		printf "%.1f &", 100*$NF;
	}
	c+=1;
}
