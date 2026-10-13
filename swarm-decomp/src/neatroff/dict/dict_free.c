#include <stdlib.h>
#include "dict.h"

void dict_free(void *dv)
{
	struct dict *d = dv;
	int i;

	if (d->dup)
		for (i = 0; i < d->cnt; i++)
			free(d->key[i]);
	free(d->hash);
	free(d->link);
	free(d->val);
	free(d->key);
	free(d);
}
