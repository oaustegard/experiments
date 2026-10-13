#include "mem_extend.h"

static void mem_extend(struct mem *m)
{
	long n = m->cap;
	char *old = m->buf;

	n = n ? 2 * n : 0x200;
	m->cap = n;
	m->buf = malloc(n);
	if (m->len)
		memcpy(m->buf, old, m->len);
	free(old);
}
