#include "mem_extend.h"

void mem_putc(struct mem *m, int c)
{
	if (m->len + 2 >= m->cap)
		mem_extend(m);
	m->buf[m->len++] = c;
}
