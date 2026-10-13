#include <string.h>
#include "sbuf.h"

void sbuf_rep(struct sbuf *sb, int c, int r)
{
	char *p = sb->buf;
	char *end = p + sb->len;

	while (p < end && (p = memchr(p, c, end - p)) != NULL) {
		*p++ = r;
	}
}
