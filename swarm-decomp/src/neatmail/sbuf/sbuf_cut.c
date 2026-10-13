#include "sbuf.h"

void sbuf_cut(struct sbuf *s, int n)
{
	if (s->len > n)
		s->len = n;
}
