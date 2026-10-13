#include "sbuf.h"

int sbuf_empty(struct sbuf *b) {
	return b->len == 0;
}
