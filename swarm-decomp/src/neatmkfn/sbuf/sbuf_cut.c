#include "sbuf.h"

void sbuf_cut(struct sbuf *sb, int n)
{
	if (sb->n > n)
		sb->n = n;
}
