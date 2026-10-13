#include <string.h>
#include "dict.h"
#include "fn_xmalloc.h"
#include "dict_extend.h"

void dict_put(struct dict *d, char *s, int v)
{
	unsigned int h = *(unsigned char *)s;
	unsigned char *p;
	int k, i, n;

	for (p = (unsigned char *)s + 1; *p; p++)
		h = h * 33 + *p;
	k = h % d->nhash;
	if (d->cnt >= d->cap)
		dict_extend(d, d->cnt + 0x400);
	if (d->dup) {
		n = strlen(s) + 1;
		s = memcpy(xmalloc(n), s, n);
	}
	i = d->cnt++;
	d->key[i] = s;
	d->val[i] = v;
	d->link[i] = d->hash[k];
	d->hash[k] = i;
}
