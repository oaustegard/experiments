static int refer_refmac(char *s, char *pat)
{
	char *p;
	int r = 0;

	if (s != NULL) {
		p = strstr(s, pat);
		if (*pat == 0 || p == NULL) {
			r = 0;
			return r;
		}
		if (p == s || p[-1] == ',') {
			unsigned char c = p[strlen(pat)];
			r = (c == ',') | (c == 0);
			return r;
		}
	}
	return r;
}
