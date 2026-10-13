#include "ic_ent.h"
#include "ic_num.h"
#include "ic_sym.h"

int ic_sym(struct ic_ent *tab, long idx, long *a, long *b)
{
	struct ic_ent *e = &tab[idx];
	long op = e->op;
	long k = op & 0xffffff;
	long t;
	int r = 1;

	if ((op & 0x400400) == 0x400400) {
		*a = e->a;
		*b = e->b;
		r = 0;
		goto done;
	}
	switch (k) {
	case 0x10:
		if (ic_sym(tab, e->a, a, b) || ic_num(tab, e->b, &t)) {
			if (ic_sym(tab, e->b, a, b)) {
				r = 1;
				goto done;
			}
			if (ic_num(tab, e->a, &t)) {
				r = 1;
				goto done;
			}
		}
		*b += t;
		r = 0;
		goto done;
	case 0x11:
		if (ic_sym(tab, e->a, a, b)) {
			r = 1;
			goto done;
		}
		if (ic_num(tab, e->b, &t)) {
			r = 1;
			goto done;
		}
		*b -= t;
		r = 0;
		goto done;
	}
done:
	return r;
}
