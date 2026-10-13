#include "mem_extend.h"

char *mem_get(struct mem *m)
{
	char *r;

	if (!m->buf)
		mem_extend(m);
	r = m->buf;
	*m = (struct mem){ 0 };
	return r;
}
