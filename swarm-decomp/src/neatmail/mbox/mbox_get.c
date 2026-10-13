#include "mbox.h"

int mbox_get(struct mbox *mp, int i, char **s, long *n)
{
	char *p;

	if (i < 0 || mp->nc <= i)
		return 1;
	p = mp->c258[i];
	if (p) {
		*s = p;
		*n = mp->c260[i];
	} else {
		*s = mp->c248[i];
		*n = mp->c250[i];
	}
	return 0;
}
