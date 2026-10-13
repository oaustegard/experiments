#include "sbuf.h"

void
sbuf_add(struct sbuf *b, int c)
{
	char *old;
	int n;

	if (b->len + 2 >= b->size) {
		n = b->size * 2;
		if (n <= 0)
			n = 1;
		n = (n + 511) & ~511;
		b->size = n;
		old = b->s;
		b->s = malloc(n);
		if (b->len)
			memcpy(b->s, old, b->len);
		free(old);
	}
	b->s[b->len++] = c;
}
