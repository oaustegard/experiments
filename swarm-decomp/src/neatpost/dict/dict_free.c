#include "dict.h"
#include "iset.h"
#include "fn_iset_free.h"
#include "fn_mextend.h"
#include <stdlib.h>

void dict_free(struct dict *d)
{
	int i;

	if (d->own) {
		for (i = 0; i < d->nkey; i++)
			free(d->keys[i]);
	}
	free(d->vals);
	free(d->keys);
	iset_free(d->set);
	free(d);
}
