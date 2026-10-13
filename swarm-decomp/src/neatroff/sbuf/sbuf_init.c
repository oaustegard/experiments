#include "fn_mextend.h"
#include "sbuf.h"

void sbuf_init(struct sbuf *b) {
	*b = (struct sbuf){0};
	b->cap = 512;
	b->buf = mextend(NULL, 0, 512, 1);
}
