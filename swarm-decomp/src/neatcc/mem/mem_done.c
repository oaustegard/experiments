#include "mem.h"

void mem_done(struct mem *m)
{
	free(m->buf);
	*m = (struct mem){ 0 };
}
