#include "sbuf.h"

char *sbuf_out(struct sbuf *b) {
	char *s = b->buf;
	*b = (struct sbuf){0};
	return s;
}
