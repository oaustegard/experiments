#include "sbuf.h"

int sbuf_len(struct sbuf *sb)
{
	return sb->n;
}
