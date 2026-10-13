static void nf_roman(char *buf, int n, char *ones, char *fives)
{
	int d, d5, i;

	while (n) {
		d5 = n % 5;
		d = n % 10;
		if (d5 == 4) {
			*buf++ = d == 9 ? ones[1] : fives[0];
			*buf++ = ones[0];
		} else {
			for (i = 0; i < d5; i++)
				*buf++ = ones[0];
			if (d > 4)
				*buf++ = fives[0];
		}
		*buf = 0;
		fives++;
		ones++;
		n /= 10;
	}
}
