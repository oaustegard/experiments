#include "ic_num.h"

int o_popnum(long *x)
{
	long n = iv_n - 1;

	if (ic_num(ic, iv[n], x))
		return 1;
	iv_n = n < 0 ? 0 : n;
	return 0;
}
