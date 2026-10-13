#include "fn_mextend.h"
#include "libc.h"
#include "sbuf.h"

void sbuf_append(struct sbuf *b, char *s) {
	int n = strlen(s);
	if (b->len + n + 1 >= b->cap) {
		b->cap = (b->len + n + 512) & ~511;
		b->buf = mextend(b->buf, b->len, b->cap, 1);
	}
	memcpy(b->buf + b->len, s, n);
	b->len += n;
}
