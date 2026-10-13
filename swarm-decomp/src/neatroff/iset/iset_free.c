#include "iset.h"
#include "fn_free.h"

void iset_free(void *set)
{
	struct iset *s = set;
	int i;

	for (i = 0; i < s->n; i++)
		free(s->tab[i]);
	free(s->tab);
	free(s->rowlen);
	free(s->rowcap);
	free(s);
}
