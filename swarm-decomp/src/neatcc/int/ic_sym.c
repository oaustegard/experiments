#include "ic_ent.h"
#include "ic_num.h"
#include "ic_sym.h"

int ic_sym(struct ic_ent *tab, long idx, long *a, long *b)
{
	struct ic_ent *e = &tab[idx];
	long k = e->op & 0xffffff;
	long t;

	if ((e->op & 0x400400) == 0x400400) {
		*a = e->a;
		*b = e->b;
		return 0;
	}
	if (k == 0x10) {
		if (ic_sym(tab, e->a, a, b) || ic_num(tab, e->b, &t)) {
			if (ic_sym(tab, e->b, a, b))
				return 1;
			if (ic_num(tab, e->a, &t))
				return 1;
		}
		*b += t;
		return 0;
	}
	if (k == 0x11) {
		if (ic_sym(tab, e->a, a, b))
			return 1;
		if (ic_num(tab, e->b, &t))
			return 1;
		*b -= t;
		return 0;
	}
	return 1;
}
