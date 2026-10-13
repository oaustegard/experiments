static int char_type(const unsigned char *s)
{
	int c = *s;
	int t = 0x12;

	if (!isdigit(c)) {
		t = 0x13;
		if (c == '"')
			return t;
		t = def_type((const char *)s);
		if (t >= 0)
			return t;
		t = 0x14;
		if ((c & 0xdf) != '^') {
			t = 0x11;
			if (ispunct(c)) {
				t = 0x10;
				if (c == '\\')
					t += (s[1] != 0);
			}
		}
	}
	return t;
}
