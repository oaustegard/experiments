static char *bracsign(const char *s, int open)
{
	char *r = (char *)s;

	switch (*s) {
	case 'c':
		if (strncmp("ceiling", s, strlen(s) < 8 ? strlen(s) : 8) == 0) {
			if (open)
				return "\\(lc";
			return "\\(rc";
		}
		break;
	case 'f':
		if (strncmp("floor", s, strlen(s) < 6 ? strlen(s) : 6) == 0) {
			if (open)
				return "\\(lf";
			return "\\(rf";
		}
		break;
	case '<':
		if (s[1] == 0)
			r = "\\(la";
		break;
	case '>':
		if (s[1] == 0)
			r = "\\(ra";
		break;
	}
	return r;
}
