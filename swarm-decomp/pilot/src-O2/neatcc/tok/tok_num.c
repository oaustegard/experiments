long tok_num(char *s, long *v)
{
	int base = 10;
	long flags = 0x14;
	long n;
	char *p;
	int c;
	int e;

	if (*s == '0' && tolower(s[1]) == 'x') {
		flags &= ~0x10;
		base = 16;
		s += 2;
	}
	if (strchr(digs, tolower((unsigned char)*s))) {
		n = 0;
		if (base == 10 && *s == '0')
			base = 8;
		while (*s && (p = strchr(digs, tolower((unsigned char)*s)))) {
			n = base * n;
			n += p - digs;
			s++;
		}
		*v = n;
		while (*s) {
			c = tolower((unsigned char)*s);
			if (c != 'u' && c != 'l')
				break;
			if (c == 'u')
				flags &= ~0x10;
			if (c == 'l')
				flags = (flags & 0x10) | 8;
			s++;
		}
		return flags;
	} else if (*s == '\'') {
		esc_char(&e, s + 1);
		*v = e;
		return flags;
	}
	return 0;
}
