char *escarg(const char *s)
{
	static char buf[256];

	if (s[1] == '\0')
		sprintf(buf, "%c", s[0]);
	else if (s[2] == '\0')
		sprintf(buf, "(%c%c", s[0], s[1]);
	else
		sprintf(buf, "[%s]", s);
	return buf;
}
