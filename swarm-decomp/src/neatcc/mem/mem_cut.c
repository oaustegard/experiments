#include "mem.h"

void mem_cut(struct mem *m, long n)
{
	m->len = m->len > n ? n : m->len;
}
