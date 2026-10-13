#include "dict.h"

int dict_idx(struct dict *d, char *s);

int dict_get(struct dict *d, char *s)
{
	int i = dict_idx(d, s);
	if (i >= 0)
		return d->val[i];
	return -1;
}
