#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_chr(struct sbuf *b, int c)
{
	char *old;
	int n, ns;

	n = b->len;
	if (n + 2 >= b->size) {
		ns = 2 * b->size;
		if (b->size >= ns)
			ns = (b->size + 128) & ~127;
		else
			ns = (ns + 127) & ~127;
		old = b->buf;
		b->size = ns;
		b->buf = malloc(ns);
		if (n)
			memcpy(b->buf, old, n);
		free(old);
	}
	b->buf[b->len++] = c;
}
