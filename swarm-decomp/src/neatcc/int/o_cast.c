#include "opt.h"

void o_cast(long t)
{
	if ((t & 0xf) != 8) {
		ic_put(t << 24 | 0x400, (int)iv[--iv_n], 0, 0);
		if (opt(1))
			return io_num();
	}
}
