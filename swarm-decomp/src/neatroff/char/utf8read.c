int utf8read(char **p, char *buf)
{
	int c = (unsigned char)**p;
	int n, i;

	n = utf8len(c);
	buf[0] = c;
	for (i = 1; i < n; i++)
		buf[i] = (*p)[i];
	buf[n] = 0;
	*p += n;
	return n;
}
