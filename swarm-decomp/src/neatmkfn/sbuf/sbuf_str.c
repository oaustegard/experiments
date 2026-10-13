#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_str(struct sbuf *sb, const char *s)
{
	char *old;
	int sz, n;

	n = strlen(s);
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
	memcpy(sb->s + sb->n, s, n);
	sb->n += n;
}
