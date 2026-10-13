#include <stdio.h>
#include "mkfn_kmin.h"
#include "mkfn_dry.h"

void mkfn_kern(const char *a, const char *b, int n)
{
	if (n == 0)
		return;
	if ((n < 0 ? -n : n) < mkfn_kmin)
		return;
	if (!mkfn_dry)
		printf("kern %s\t%s\t%d\n", a, b, n);
}
