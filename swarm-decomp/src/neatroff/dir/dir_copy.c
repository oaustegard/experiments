#include <stdlib.h>
#include <string.h>
#include "fn_xmalloc.h"
#include "dir_copy.h"

static void dir_copy(char **buf, int *n, int *sz, const char *s, int len, int flag)
{
	char *nb;
	int cap = *sz;

	nb = *buf;
	if (*n + len < cap)
		goto copy;
	do {
		if (cap == 0)
			cap = 512;
		else
			cap = cap * 2;
		nb = xmalloc(cap + 1);
		if (*sz != 0)
			memcpy(flag ? nb + *sz : nb, *buf, *sz);
		free(*buf);
		*sz = cap;
		*buf = nb;
	} while (*n + len >= cap);
copy:
	if (flag > 0)
		memcpy(*buf + cap - len - *n, s, len);
	else
		memcpy(*buf + *n, s, len);
	*n += len;
}
