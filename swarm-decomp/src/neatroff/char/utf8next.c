int utf8next(char *s, int (*next)(void))
{
	int c = next();
	int n, i;

	if (c < 0)
		return 0;
	n = utf8len(c);
	s[0] = c;
	for (i = 1; i < n; i++)
		s[i] = next();
	s[n] = 0;
	return n;
}
