#include "mem_extend.h"

void mem_put(struct mem *m, char *s, long n)
{
	long off = m->len;

	while (m->len + off + n + 1 >= m->cap)
		mem_extend(m);
	memcpy(m->buf + off, s, n);
	m->len += n;
}
