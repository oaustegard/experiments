#include "mbox.h"
#include <stdlib.h>
#include <string.h>

int mbox_set(struct mbox *mp, int i, char *s, long n)
{
	char **pp;

	if (i < 0 || mp->nc <= i)
		return 1;
	free(mp->c258[i]);
	pp = &mp->c258[i];
	*pp = malloc(n + 1);
	if (*pp) {
		mp->c260[i] = n;
		memcpy(*pp, s, n);
		(*pp)[n] = 0;
	}
	return 0;
}
