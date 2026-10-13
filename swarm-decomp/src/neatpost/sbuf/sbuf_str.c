#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

int sbuf_str(struct sbuf *b, const char *p)
{
	char *old;
	int len, n, ns;

	n = strlen(p);
	len = b->len;
	if (len + n + 1 >= b->size) {
		old = b->buf;
		ns = b->size + n + 1;
		if (ns < 2 * b->size)
			ns = 2 * b->size;
		ns = (ns + 127) & ~127;
		b->size = ns;
		b->buf = malloc(ns);
		if (len)
			memcpy(b->buf, old, len);
		free(old);
		len = b->len;
	}
	memcpy(b->buf + len, p, n);
	b->len += n;
}
