#include "sbuf.h"

void
sbuf_append(struct sbuf *b, const char *s)
{
	char *old;
	int n = strlen(s);
	int len;

	if (b->len + 1 + n >= b->size) {
		int sz = b->len + 1 + n;
		if (sz <= 0)
			sz = 1;
		sz = (sz + 511) & ~511;
		b->size = sz;
		old = b->s;
		b->s = malloc(sz);
		if (b->len)
			memcpy(b->s, old, b->len);
		free(old);
	}
	len = b->len;
	memcpy(b->s + len, s, n);
	b->len += n;
}
