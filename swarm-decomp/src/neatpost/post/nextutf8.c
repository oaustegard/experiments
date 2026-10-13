static void nextutf8(char *s, int unused)
{
	int c, n, i;

	c = getc(stdin);
	if ((c & 0xc0) == 0xc0) {
		if (!(c & 0x20)) {
			if (c < 0)
				return;
			n = 2;
		} else if (!(c & 0x10)) {
			if (c < 0)
				return;
			n = 3;
		} else if (!(c & 8)) {
			if (c < 0)
				return;
			n = 4;
		} else {
			if (c < 0)
				return;
			n = 1;
		}
	} else {
		if (c < 0)
			return;
		n = c > 0;
	}
	*s = c;
	for (i = 1; i < n; i++)
		s[i] = getc(stdin);
	s[n] = 0;
}
