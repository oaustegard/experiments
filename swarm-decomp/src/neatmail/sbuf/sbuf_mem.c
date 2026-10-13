#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_mem(struct sbuf *sb, const char *s, int n)
{
	int len = sb->len;

	if (len + n + 1 >= sb->size) {
		char *old = sb->buf;
		int size = sb->size + n + 1;

		if (size < 2 * sb->size)
			size = 2 * sb->size;
		size = (size + 0x7f) & ~0x7f;
		sb->size = size;
		sb->buf = malloc(size);
		if (len)
			memcpy(sb->buf, old, len);
		free(old);
		len = sb->len;
	}
	memcpy(sb->buf + len, s, n);
	sb->len += n;
}
