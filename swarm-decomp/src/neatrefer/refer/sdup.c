static char *sdup(char *s)
{
	char *e = strchr(s, '\n');
	int n;
	char *d;

	if (e == NULL)
		e = s + strlen(s);
	n = e - s;
	d = malloc(n + 1);
	memcpy(d, s, n);
	d[n] = 0;
	return d;
}
