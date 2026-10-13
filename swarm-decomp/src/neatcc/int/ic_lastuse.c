#include "ic_ent.h"

/* for each entry, the index of the last entry that uses it; -1 for none */
long *ic_lastuse(struct ic_ent *tab, long n)
{
	long *use = calloc(n, 8);
	int i = n - 1;
	int k;
	long op, d, cnt, j;
	struct ic_ent *e;

	for (; i >= 0; i--) {
		e = &tab[i];
		k = ic_regcnt(e);
		op = e->op;
		d = op & 0x200;
		if (!use[i]) {
			if (op & 0x107f0) {
				if (!d)
					continue;
				use[i] = -1;
			} else
				use[i] = -1;
		}
		if (k > 0) {
			if (!use[e->a])
				use[e->a] = i;
		}
		if (k > 1) {
			if (!use[e->b])
				use[e->b] = i;
		}
		if (k > 2) {
			if (!use[e->c])
				use[e->c] = i;
		}
		if (d) {
			cnt = e->c;
			for (j = 0; j < cnt; j++) {
				long v = ((long *)e->d)[j];

				if (!use[v])
					use[v] = i;
			}
		}
	}
	return use;
}
