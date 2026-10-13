#include <stdlib.h>
#include "iset.h"

void iset_free(struct iset *s)
{
	int i;

	for (i = 0; i < s->n; i++)
		free(s->a[i]);
	free(s->a);
	free(s->c);
	free(s->b);
	free(s);
}
