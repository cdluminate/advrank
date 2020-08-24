/Summary/{
	count++;
	if (count % 2 == 1) {
		printf "50 $\\rightarrow$ %.1f & 50 $\\rightarrow$ %.1f & ", $4 * 100, $6 * 100;
	} else {
		printf "%.1f $\\rightarrow$ %.1f & %.1f $\\rightarrow$ %.1f & ",
			   $8*100, $4*100, $10*100, $6*100;
	}
}
