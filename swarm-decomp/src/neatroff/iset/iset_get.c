#include "iset.h"

int *iset_get(void *set, int i)
{
	struct iset *s = set;
	int *p = 0;

	if (i >= 0 && i < s->n)
		p = s->tab[i];
	return p;
}
