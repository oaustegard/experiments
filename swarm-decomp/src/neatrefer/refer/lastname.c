static char *lastname(char *s)
{
	char *p = s;
	char *r = s;
	unsigned char c = *s;
	unsigned char a = c;

	if (c == 0)
		return r;
	for (;;) {
		if (!islower(c))
			r = p;
		while (a & 0xdf) {
			if (a == '\\') {
				a = p[2];
				p += 2;
			} else {
				p++;
				a = *p;
			}
		}
		if (a != ' ')
			return r;
		do {
			a = p[1];
			p++;
		} while (a == ' ');
		if (a == 0)
			return r;
		c = *r;
	}
}
