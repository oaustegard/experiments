#include "mbox.h"

static int mbox_find(struct mbox *m, int i)
{
	int a;
	for (a = m->start[i]; a < m->end[i]; a++)
		if (m->c258[a])
			return 1;
	return 0;
}
