#include "dict.h"
#include "fn_mextend.h"

static void dict_extend(struct dict *d, int n)
{
	d->link = mextend(d->link, d->cap, n, sizeof(int));
	d->key = mextend(d->key, d->cap, n, sizeof(char *));
	d->val = mextend(d->val, d->cap, n, sizeof(int));
	d->cap = n;
}
