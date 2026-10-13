#include "dict.h"
#include "fn_iset_get.h"
#include "fn_iset_len.h"
#include <string.h>

int dict_idx(struct dict *d, const char *key)
{
	unsigned long h;
	const char *p, *e;
	int *r, *q;
	int b, len;

	h = (unsigned char)*key;
	p = key + 1;
	if ((int)d->maxlen > 1) {
		e = key + (unsigned int)d->maxlen;
		while (*p) {
			h = h * 33 + (unsigned char)*p;
			p++;
			if (p == e)
				break;
		}
	}
	b = h & 0x3ff;
	r = iset_get(d->set, b);
	len = iset_len(d->set, b);
	q = r + len;
	if (r) {
		while (--q >= r) {
			int i = *q;

			if (strcmp(d->keys[i], key) == 0)
				return i;
		}
	}
	return -1;
}
