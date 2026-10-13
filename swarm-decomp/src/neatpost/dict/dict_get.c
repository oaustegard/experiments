#include "dict.h"
#include "fn_dict_idx.h"

int dict_get(struct dict *d, const char *key)
{
	int i;

	i = dict_idx(d, key);
	if (i < 0)
		return d->dflt;
	return d->vals[i];
}
