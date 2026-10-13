#include <stdlib.h>
#include "dict.h"
#include "fn_xmalloc.h"
#include "dict_extend.h"

/*
 * The header fields are cleared in four blocks, in this order: val/cap/cnt
 * (16 bytes at 0x18), nhash/dup (8 bytes at 0x28), cnt = 1, then link/key
 * (16 bytes at 0x08). The blocks are written as whole aggregates so gcc keeps
 * them as wide stores instead of dropping the ones that are overwritten.
 */
struct dict_vcc { void *val; int cap; int cnt; };
struct dict_nd { int nhash; int dup; };
struct dict_lk { int *link; char **key; };

struct dict *dict_make(int n, int dup)
{
	struct dict *d = xmalloc(sizeof(struct dict));
	*(struct dict_vcc *)&d->val = (struct dict_vcc){ 0 };
	*(struct dict_nd *)&d->nhash = (struct dict_nd){ 0 };
	d->cnt = 1;
	*(struct dict_lk *)&d->link = (struct dict_lk){ 0 };
	d->hash = calloc(n, sizeof(int));
	d->nhash = n;
	d->dup = dup;
	dict_extend(d, 0x400);
	return d;
}
