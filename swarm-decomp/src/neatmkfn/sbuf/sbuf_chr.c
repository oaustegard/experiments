#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_chr(struct sbuf *sb, int c)
{
	char *old;
	int sz, d;

	if (sb->n + 2 >= sb->sz) {
		old = sb->s;
		d = 2 * sb->sz;
		if (sb->sz >= d)
			sz = (sb->sz + 128) & ~127;
		else
			sz = (d + 127) & ~127;
		sb->sz = sz;
		sb->s = malloc(sz);
		if (sb->n)
			memcpy(sb->s, old, sb->n);
		free(old);
	}
	sb->s[sb->n++] = c;
}
