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

	if (k >= 0) {
		memcpy(buf, &iv[iv_n - n], 8 * (long)n);
		iv_n -= n;
		for (; k >= 0; k--) {
			long v = buf[k];

			if (!opt(1))
				continue;
			if ((ic[(int)v].op & 0x400 && ic[(int)v].op & 0x700000) || ic_load((int)v)) {
				ic_put(ic[(int)v].op, ic[(int)v].a, ic[(int)v].b, ic[(int)v].c);
				buf[k] = iv[--iv_n];
			}
		}
	}
	ic_put(0x200, (int)iv[--iv_n], 0, n);
	ic[ic_n - 1].d = (char *)buf;
	if (!m)
		iv_n--;
	iv_n = iv_n < 0 ? 0 : iv_n;
	if (opt(1)) {
		struct ic_ent *e = &ic[ic_n - 1];
		long a, b;

		if (e->op & 0x200) {
			if (!ic_sym(ic, e->a, &a, &b) && !b) {
				e->op |= 0x400000;
				e->a = a;
			}
		}
	}
}
