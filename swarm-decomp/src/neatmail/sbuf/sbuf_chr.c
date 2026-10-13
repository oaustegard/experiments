#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void sbuf_chr(struct sbuf *sb, int c)
{
	if (sb->len + 2 >= sb->size) {
		char *old = sb->buf;
		int size;

		if (sb->size >= 2 * sb->size)
			size = (sb->size + 0x80) & ~0x7f;
		else
			size = (2 * sb->size + 0x7f) & ~0x7f;
		sb->size = size;
		sb->buf = malloc(size);
		if (sb->len)
			memcpy(sb->buf, old, sb->len);
		free(old);
	}
	sb->buf[sb->len++] = c;
}
