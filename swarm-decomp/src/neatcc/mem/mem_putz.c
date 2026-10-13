#include "mem_extend.h"

void mem_putz(struct mem *m, long n)
{
	while (m->len + n + 1 >= m->cap)
		mem_extend(m);
	memset(m->buf + m->len, 0, n);
	m->len += n;
}
