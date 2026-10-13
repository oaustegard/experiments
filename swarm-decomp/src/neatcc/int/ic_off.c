#include "ic_ent.h"
#include "ic_num.h"
#include "ic_off.h"

/* resolve entry idx of tab into *a and *b (offsets); nonzero on failure */
static int ic_off(struct ic_ent *tab, long idx, long *a, long *b)
{
	struct ic_ent *e = &tab[idx];
	long k = e->op & 0xffffff;
	long t;

	if (k == 0x100010) {
		*a = e->a;
		*b = e->b;
		return 0;
	}
	if (k == 0x100011) {
		*a = e->a;
		*b = -e->b;
		return 0;
	}
	if (k == 0x10) {
		if (ic_off(tab, e->a, a, b) || ic_num(tab, e->b, &t)) {
			if (ic_off(tab, e->b, a, b))
				return 1;
			if (ic_num(tab, e->a, &t))
				return 1;
		}
		*b += t;
		return 0;
	}
	if (k == 0x11) {
		if (ic_off(tab, e->a, a, b))
			return 1;
		if (ic_num(tab, e->b, &t))
			return 1;
		*b -= t;
		return 0;
	}
	*a = idx;
	*b = 0;
	return 0;
}
