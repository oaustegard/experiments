#include "iset.h"

int iset_has(void *set, int i, int v)
{
	struct iset *s = set;
	int *p;
	int n;

	if (i < 0 || i >= s->n)
		return 0;
	n = s->rowlen[i];
	if (n <= 0)
		return 0;
	for (p = s->tab[i]; p != s->tab[i] + n; p++)
		if (*p == v)
			return 1;
	return 0;
}
