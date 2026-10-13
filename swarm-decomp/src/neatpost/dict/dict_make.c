#include "dict.h"
#include "fn_iset_make.h"
#include "fn_mextend.h"
#include <stdlib.h>

struct dict *dict_make(int dflt, int own, int maxlen)
{
	struct dict *d;
	int m;

	d = calloc(sizeof *d, 1);
	d->used1c = 1;
	m = maxlen ? maxlen : 0x20;
	d->own = own;
	d->dflt = dflt;
	d->maxlen = m;
	d->set = iset_make();
	d->keys = mextend(d->keys, d->nkey, 0x400, sizeof(char *));
	d->vals = mextend(d->vals, d->nkey, 0x400, sizeof(int));
	d->nkey = 0x400;
	return d;
}
