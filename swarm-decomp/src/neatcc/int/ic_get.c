#include "ic_ent.h"
#include "ic_tab.h"
#include "ic_put.h"
#include "iv_tab.h"
#include "lab_tab.h"
#include "fn_ic_regcnt.h"

/* take the int-code table and its count out of the globals into *pic and *pn */
void ic_get(struct ic_ent **pic, long *pn)
{
	struct ic_ent *tab;
	struct ic_ent *e;
	long *lab;
	long *remap;
	char *mark;
	long n, nn, i, j, op, d;
	int k;

	n = ic_n;
	if (n == 0 || !(ic[n - 1].op & 0x8000) || n == lab_last) {
		ic_put(0x100400, 0, 0, 0);
		ic_put(0x8000, iv[--iv_n], 0, 0);
	}
	tab = ic;
	n = ic_n;
	lab = lab_loc;
	for (i = 0; i < n; i++)
		if (tab[i].op & 0x7000)
			tab[i].c = lab[tab[i].c];
	mark = calloc(n, 1);
	for (i = n - 1; i >= 0; i--) {
		e = &tab[i];
		k = ic_regcnt(e);
		op = e->op;
		d = op & 0x200;
		if (!(op & 0x107f0) || d)
			mark[i] = 1;
		else if (!mark[i])
			continue;
		if (k > 0)
			mark[e->a] = 1;
		if (k > 1)
			mark[e->b] = 1;
		if (k > 2)
			mark[e->c] = 1;
		if (d) {
			for (j = 0; j < e->c; j++)
				mark[((long *)e->d)[j]] = 1;
		}
	}
	remap = calloc(n, 8);
	nn = 0;
	for (i = 0; i < n; i++) {
		remap[i] = nn;
		if (mark[i]) {
			if (nn != i)
				tab[nn] = tab[i];
			nn++;
		} else if (tab[i].op & 0x200)
			free(tab[i].d);
	}
	ic_n = nn;
	for (i = 0; i < nn; i++) {
		e = &tab[i];
		k = ic_regcnt(e);
		if (k > 0) {
			e->a = remap[e->a];
			if (k > 1) {
				e->b = remap[e->b];
				if (k > 2)
					e->c = remap[e->c];
			}
		}
		if (e->op & 0x7000)
			e->c = remap[e->c];
		if (e->op & 0x200) {
			for (j = 0; j < e->c; j++)
				((long *)e->d)[j] = remap[((long *)e->d)[j]];
		}
	}
	free(mark);
	free(remap);
	ic = 0;
	ic_n = 0;
	*pic = tab;
	ic_sz = 0;
	*pn = nn;
	iv_n = 0;
	free(lab);
	lab_loc = 0;
	lab_n = 0;
	lab_sz = 0;
	lab_last = 0;
}
