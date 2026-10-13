#include "dict.h"

char *dict_key(struct dict *d, int i)
{
	return d->key[i];
}
