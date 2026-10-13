#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

char *sbuf_done(struct sbuf *b)
{
	char *p;
	int n;

	p = b->buf;
	if (p == NULL) {
		b->size = 1;
		p = malloc(1);
		b->buf = p;
		n = 0;
		if (b->len) {
			n = b->len;
			memcpy(p, 0, n);
		}
	} else {
		n = b->len;
	}
	p[n] = 0;
	p = b->buf;
	free(b);
	return p;
}
