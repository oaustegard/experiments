#include "iset.h"

int iset_len(void *set, int i)
{
	struct iset *s = set;

	if (i < 0 || i >= s->n)
		return 0;
	return s->rowlen[i];
}
