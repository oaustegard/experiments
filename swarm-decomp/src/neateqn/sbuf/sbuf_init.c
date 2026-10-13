#include "sbuf.h"

void
sbuf_init(struct sbuf *b)
{
	b->len = 0;
	b->s = malloc(512);
	b->size = 512;
}
