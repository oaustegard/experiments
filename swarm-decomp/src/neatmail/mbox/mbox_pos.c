#include "mbox.h"

int mbox_pos(struct mbox *m, int x, int *pos)
{
	int i, k = 0;
	int s;

	for (i = 1; i < m->n; i++) {
		k = i;
		if (m->start[i] > x) {
			k = i - 1;
			break;
		}
	}
	s = m->start[k];
	*pos = x - s;
	return k;
}
