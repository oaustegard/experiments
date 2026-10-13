#include "ic_ent.h"
#include "ic_tab.h"
#include "ic_put.h"
#include "ic_load.h"
#include "iv_tab.h"
#include "opt.h"
#include "io_loc.h"

void o_assign(long x)
{
	long t = iv[iv_n - 1];
	int i;

	iv_n -= 2;
	i = iv[iv_n];
	if (opt(1)) {
		if ((ic[i].op & 0x400 && ic[i].op & 0x700000) || ic_load(i)) {
			ic_put(ic[i].op, ic[i].a, ic[i].b, ic[i].c);
			iv_n--;
			i = iv[iv_n];
		}
	}
	ic_put(x << 24 | 0x120000, (int)t, i, 0);
	iv[iv_n++] = (int)t;
	if (opt(1))
		io_loc();
}
