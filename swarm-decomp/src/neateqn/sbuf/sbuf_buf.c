#include "sbuf.h"

char *
sbuf_buf(struct sbuf *b)
{
	b->s[b->len] = '\0';
	return b->s;
}
