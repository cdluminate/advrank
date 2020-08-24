/transfer.*Original/{
	c+=1;x+=$6;
}

/Summary/{
	if (nsum % 5 == 0) {
		print "";
		print "";
		}
	if (nsum % 6 == 0) {
		printf "%.1f$\\rightarrow$%.1f & ", 100*x/c, 100*$NF;
	} else {
		printf "%.1f & ", 100*x/c, 100*$NF;
	}
   	c=0; x=0;
	nsum+=1;
}
