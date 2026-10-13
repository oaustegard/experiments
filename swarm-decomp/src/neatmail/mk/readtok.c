static char *readtok(unsigned char *s, char *buf) {
	unsigned char c;

	c = *s;
	while (c <= 58 && ((0x0400010100000600ULL >> c) & 1)) {
		if (c == '(') {
			do {
				c = *++s;
				if (c == 0) {
					*buf = 0;
					return (char *)s;
				}
			} while (c != ')');
		}
		c = *++s;
	}
	if (c == 0) {
		*buf = 0;
		return (char *)s;
	}
	while (!(c <= 58 && ((0x0400010000000600ULL >> c) & 1))) {
		*buf++ = c;
		c = *++s;
		if (c == 0 || c == ' ')
			break;
	}
	*buf = 0;
	return (char *)s;
}
