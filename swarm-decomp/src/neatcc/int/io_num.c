#include "ic_num.h"

static int io_num(void)
{
	long x;
	long n = iv_n - 1;
	int r;

	r = ic_num(ic, iv[n], &x);
	if (r)
		return r;
	iv_n = n < 0 ? 0 : n;
	ic_put(0x100400, x, 0, 0);
}
