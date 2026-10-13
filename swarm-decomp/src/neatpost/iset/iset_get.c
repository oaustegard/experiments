#include "iset.h"
#include "fn_iset_get.h"

int *iset_get(struct iset *s, int i)
{
	int *p = 0;

	if (i >= 0 && i < s->n)
		p = s->a[i];
	return p;
}
