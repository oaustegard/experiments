#include "mem_extend.h"

void mem_put(struct mem *m, char *s, long n)
{
	long off = m->len;

	for (;;) {
		long len = m->len;

		if (len + off + n + 1 < m->cap)
			break;
		mem_extend(m);
	}
	memcpy(m->buf + off, s, n);
	m->len += n;
}
