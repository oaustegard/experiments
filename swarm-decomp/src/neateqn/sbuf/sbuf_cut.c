#include "sbuf.h"

void
sbuf_cut(struct sbuf *b, int n)
{
	if (b->len > n)
		b->len = n;
}
