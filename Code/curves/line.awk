/Summary/{
	if (count > 0 && count <= 8) {
		printf " %.1f &", $6 * 100;
	} else if (count > 8 && count <=16) {
		printf " %.1f,~%.1f &", $6 * 100, $14 * 100;
	}
	count++;
}
END{
	print "";
}
