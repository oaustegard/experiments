int mbox_ith(char *path, int ith, char **buf, long *len)
{
	char *p, *q;
	int i, fd, size;
	char *dat, *end;

	fd = open(path, 0);
	if (fd < 0)
		return 1;
	size = filesize(fd);
	dat = malloc(size + 1);
	if (!dat)
		return 1;
	xread(fd, dat, size);
	dat[size] = 0;
	close(fd);
	end = dat + size;
	p = mbox_from_(dat, end);
	for (i = 0; p && i < ith; i++)
		p = mbox_from_(p + 1, end);
	if (!p)
		return 1;
	q = mbox_from_(p + 1, end);
	if (!q)
		q = dat + size;
	*buf = malloc(q - p + 1);
	if (!*buf)
		return 1;
	*len = q - p;
	memcpy(*buf, p, *len);
	(*buf)[*len] = 0;
	free(dat);
	return 0;
}
