#include "iset.h"
#include "fn_iset_len.h"

int iset_len(struct iset *s, int i)
{
	if (i < 0 || i >= s->n)
		return 0;
	return s->c[i];
}
