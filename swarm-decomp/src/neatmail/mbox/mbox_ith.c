#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "fn_xread.h"

int mbox_ith(char *name, int i, char **s, long *n)
{
	struct stat st;
	int fd, size, j;
	char *buf, *end, *p, *q, *t;
	long len;

	fd = open(name, O_RDONLY);
	if (fd < 0)
		goto fail;
	fstat(fd, &st);
	size = st.st_size;
	buf = malloc(size + 1);
	if (!buf)
		goto fail;
	xread(fd, buf, size);
	end = buf + size;
	buf[size] = 0;
	close(fd);
	p = mbox_from_(buf, end);
	if (!p)
		goto fail;
	for (j = 0; p && j < i; j++)
		p = mbox_from_(p + 1, end);
	if (!p)
		goto fail;
	q = mbox_from_(p + 1, end);
	if (!q)
		q = end;
	len = q - p;
	t = malloc(len + 1);
	*s = t;
	if (!t)
		goto fail;
	*n = len;
	memcpy(t, p, len);
	t[len] = 0;
	free(buf);
	return 0;
fail:
	return 1;
}
