#include <string.h>

void sbuf_append(struct sbuf *sb, const char *s)
{
	int n = strlen(s);

	if (sb->n + n + 1 >= sb->sz)
		sbuf_extend(sb, sb->n + n + 1);
	memcpy(sb->s + sb->n, s, n);
	sb->n += n;
}
