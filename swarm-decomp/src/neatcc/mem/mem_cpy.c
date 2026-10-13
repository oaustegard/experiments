#include "mem_extend.h"

void mem_cpy(struct mem *m, long off, char *src, long n)
{
	while (off + m->len + n + 1 >= m->cap)
		mem_extend(m);
	memcpy(m->buf + off, src, n);
}
