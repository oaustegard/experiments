#include "dict.h"
#include "fn_iset_put.h"
#include "fn_mextend.h"
#include <stdlib.h>
#include <string.h>

void dict_put(struct dict *d, const char *key, int val)
{
	unsigned long h;
	const char *p, *e;
	int idx, nn;
	const char *kk = key;	/* hash reads the caller's copy; key may be reassigned below */

	if (d->used1c >= d->nkey) {
		nn = d->used1c + 0x400;
		d->keys = mextend(d->keys, d->nkey, nn, sizeof(char *));
		d->vals = mextend(d->vals, d->nkey, nn, sizeof(int));
		d->nkey = nn;
	}
	if (d->own) {
		int len = strlen(key) + 1;
		char *q = malloc(len);

		memcpy(q, key, len);
		key = q;
	}
	idx = d->used1c;
	d->used1c = idx + 1;
	d->keys[idx] = (char *)key;
	d->vals[idx] = val;

	h = (unsigned char)*kk;
	p = kk + 1;
	if (d->maxlen > 1) {
		e = key + (unsigned int)d->maxlen;
		while (*p) {
			h = h * 33 + (unsigned char)*p;
			p++;
			if (p == e)
				break;
		}
	}
	iset_put(d->set, h & 0x3ff, idx);
}
