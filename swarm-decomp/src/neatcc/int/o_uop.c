#include "opt.h"

void o_uop(long t)
{
	long n;
	long a;
	struct ic_ent *e;

	ic_put(t, (int)iv[--iv_n], 0, 0);
	if (opt(1)) {
		io_num();
		n = iv_n - 1;
		e = &ic[iv[n]];
		a = e->a;
		if ((e->op & 0xffffff) == 0x102) {
			e = &ic[a];
			if (e->op & 0x80) {
				if (n < 0)
					n = 0;
				e->op ^= 1;
				iv_n = n + 1;
				iv[n] = a;
			}
		}
	}
}
