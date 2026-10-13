#include "ic_ent.h"

void ic_free(struct ic_ent *e)
{
	if (e->op & 0x200)
		free(e->d);
}
