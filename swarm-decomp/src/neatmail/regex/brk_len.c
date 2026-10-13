static int brk_len(const char *s)
{
	int n = 1;
	unsigned char c;

	if (s[1] == '^')
		n++;
	c = s[n];
	if (c == ']') {
		n++;
		c = s[n];
	}
	while (c) {
		if (c == ']')
			return n + 1;
		if (c == '[') {
			unsigned char d = s[n + 1];
			if (d == ':' || d == '=') {
				do {
					n++;
					c = s[n];
				} while (c && c != ']');
				if (!c)
					return n;
			}
		}
		n++;
		c = s[n];
	}
	return n;
}
