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
	if ((int)d->maxlen > 1) {
		e = key + (unsigned int)d->maxlen;
		while (*p) {
			h = h * 33 + (unsigned char)*p;
			p++;
			if (p == e)
				break;
		}
	}
	r = iset_get(d->set, h & 0x3ff);
	while (r) {
		int n;

		i = r[++*pn];
		if (i < 0)
			break;
		n = strlen(d->keys[i]);
		if (strncmp(d->keys[i], key, n) == 0)
			return d->vals[i];
	}
	return d->dflt;
}
