#include "dict.h"
#include "fn_iset_get.h"
#include "fn_mextend.h"
#include <string.h>

int dict_prefix(struct dict *d, const char *key, int *pn)
{
	unsigned long h;
	const char *p, *e;
	int *r;
	int i;

	p = key + 1;
	h = (unsigned char)*key;
	if (d->maxlen > 1) {
		e = key + (unsigned int)d->maxlen;
		while (*p) {
			h = h * 33 + (unsigned char)*p;
			p++;
			if (p == e)
				break;
		}
	}
	r = iset_get(d->set, h & 0x3ff);
	if (!r)
		return d->dflt;
	for (;;) {
		i = r[++*pn];
		if (i < 0)
			return d->dflt;
		if (strncmp(d->keys[i], key, strlen(d->keys[i])) == 0)
			return d->vals[i];
	}
}
