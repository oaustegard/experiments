#include "ic_ent.h"
#include "ic_tab.h"
#include "ic_put.h"
#include "ic_load.h"
#include "ic_sym.h"
#include "iv_tab.h"
#include "opt.h"

void o_call(int n, int m)
{
	long *buf = malloc(8 * (long)n);
	int k = n - 1;
	struct ic_ent *e;
	int i;

	for (i = k; i >= 0; i--)
		buf[i] = iv[--iv_n];
	for (; k >= 0; k--) {
		long v = buf[k];

		if (!opt(1))
			continue;
		if ((ic[(int)v].op & 0x400 && ic[(int)v].op & 0x700000) || ic_load((int)v)) {
			ic_put(ic[(int)v].op, ic[(int)v].a, ic[(int)v].b, ic[(int)v].c);
			buf[k] = iv[--iv_n];
		}
	}
	e = ic_put(0x200, (int)iv[--iv_n], 0, n);
	e->d = (char *)buf;
	iv_n -= !m;
	iv_n = iv_n < 0 ? 0 : iv_n;
	if (opt(1)) {
		struct ic_ent *f = &ic[ic_n - 1];
		long a, b;

		if (f->op & 0x200) {
			if (!ic_sym(ic, f->a, &a, &b) && !b) {
				f->op |= 0x400000;
				f->a = a;
			}
		}
	}
}
