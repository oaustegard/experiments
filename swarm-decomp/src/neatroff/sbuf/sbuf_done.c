#include "libc.h"
#include "sbuf.h"

void sbuf_done(struct sbuf *b) {
	free(b->buf);
}
