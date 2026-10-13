static int readchar(char *s)
{
	int c, n, k;

	c = src_next();
	s[0] = c;
	n = c > 0;
	if (c == '\\') {
		c = src_next();
		s[1] = c;
		n = 2;
		if (c == '(') {
			s[2] = src_next();
			s[3] = src_next();
			n = 4;
		} else if (c == '[') {
			k = 3;
			do {
				c = src_next();
				n = k;
				s[k - 1] = c;
				k++;
			} while (c > 0 && c != ']');
		}
	}
	return n;
}
