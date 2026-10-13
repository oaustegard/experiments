#include "sbuf.h"

char *sbuf_buf(struct sbuf *b) {
	b->buf[b->len] = '\0';
	return b->buf;
}
