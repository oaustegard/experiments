int utf8read(char **p, char *buf)
{
	int c = (unsigned char)**p;
	int n, i;

	if (~c & 0xc0) {
		n = c > 0;
		goto out;
	}
	if (!(c & 0x20)) {
		n = 2;
		goto out;
	}
	if (!(c & 0x10)) {
		n = 3;
		goto out;
	}
	n = (c & 8) ? 1 : 4;
out:
	buf[0] = c;
	for (i = 1; i < n; i++)
		buf[i] = (*p)[i];
	buf[n] = 0;
	*p += n;
	return n;
}
