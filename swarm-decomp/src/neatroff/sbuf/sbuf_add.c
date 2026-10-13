#include "fn_mextend.h"
#include "sbuf.h"

void sbuf_add(struct sbuf *b, int c) {
	if (b->len + 2 >= b->cap) {
		b->cap = (b->cap * 2 + 511) & ~511;
		b->buf = mextend(b->buf, b->len, b->cap, 1);
	}
	b->buf[b->len++] = c;
}
