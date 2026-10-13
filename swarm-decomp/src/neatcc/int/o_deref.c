#include "opt.h"
#include "io_loc.h"

void o_deref(long t)
{
	ic_put(t << 24 | 0x110000, (int)iv[--iv_n], 0, 0);
	if (opt(1))
		io_loc();
}
