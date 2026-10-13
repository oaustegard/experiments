#include "iv_tab.h"

void o_tmpswap(void)
{
	long *p = iv - 2;
	long n = iv_n;
	long t = p[n];

	p[n] = p[n + 1];
	p[n + 1] = t;
}
