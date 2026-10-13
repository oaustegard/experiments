#include "ic_sym.h"

int o_popsym(long *a, long *b)
{
	long n = iv_n - 1;

	if (ic_sym(ic, iv[n], a, b))
		return 1;
	iv_n = n < 0 ? 0 : n;
	return 0;
}
