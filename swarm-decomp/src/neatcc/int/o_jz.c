#include "ic_ent.h"
#include "ic_tab.h"
#include "mextend.h"
#include "iv_tab.h"
#include "opt.h"

void o_jz(long x)
{
	struct ic_ent *e, *s;
	long v = iv[--iv_n];
	long n, sz;
	long op, r1, r2;

	if (ic_n == ic_sz) {
		sz = 2 * ic_n;
		sz = sz < 0x80 ? 0x80 : sz;
		ic_sz = sz;
		ic = mextend(ic, ic_n, sz, sizeof(struct ic_ent));
	}
	n = ic_n;
	ic_n = n + 1;
	e = &ic[n];
	e->op = 0x2000;
	e->a = v;
	e->b = 0;
	e->c = x;
	if (opt(1))
		goto body;
	return;
body:
	e = &ic[ic_n - 1];
	if (e->op & 0x2000) {
		s = &ic[e->a];
		if ((s->op & 0xffffff) == 0x102) {
			e->op ^= 1;
			e->a = s->a;
		} else if (s->op & 0x80) {
			op = s->op & 0xffffffffffffbf7fL;
			r1 = op ^ 0x4001;
			r2 = op | 0x4000;
			if ((e->op & 0xffffff) == 0x2000)
				r2 = r1;
			e->b = s->b;
			e->op = r2;
			e->a = s->a;
		}
	}
}
