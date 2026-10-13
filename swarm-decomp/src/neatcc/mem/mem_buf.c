#include "mem.h"

char *mem_buf(struct mem *m)
{
	if (!m->buf)
		return "";
	m->buf[m->len] = 0;
	return m->buf;
}
