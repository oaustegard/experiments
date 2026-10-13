#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

char *sbuf_buf(struct sbuf *sb)
{
	char *p = sb->buf;

	if (!p) {
		sb->size = 1;
		p = sb->buf = malloc(1);
		if (sb->len)
			memcpy(p, NULL, sb->len);
	}
	p[sb->len] = 0;
	return sb->buf;
}
