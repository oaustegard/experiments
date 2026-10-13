#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_mem(struct sbuf *sb, const void *p, int n)
{
	char *old;
	int sz;

	if (sb->n + n + 1 >= sb->sz) {
		old = sb->s;
		sz = sb->sz + n + 1;
		if (sz < 2 * sb->sz)
			sz = 2 * sb->sz;
		sz = (sz + 127) & ~127;
		sb->sz = sz;
		sb->s = malloc(sz);
		if (sb->n)
			memcpy(sb->s, old, sb->n);
		free(old);
	}
	memcpy(sb->s + sb->n, p, n);
	sb->n += n;
}
