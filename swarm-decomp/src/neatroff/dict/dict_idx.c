#include <string.h>
#include "dict.h"

int dict_idx(struct dict *d, char *s)
{
	int *hash = d->hash;
	unsigned int h = 0;
	unsigned char *p;
	int k, i;

	h = *(unsigned char *)s;
	for (p = (unsigned char *)s + 1; *p; p++)
		h = h * 33 + *p;
	k = h % d->nhash;
	for (i = hash[k]; i > 0; i = d->link[i])
		if (strcmp(s, d->key[i]) == 0)
			return i;
	return -1;
}
