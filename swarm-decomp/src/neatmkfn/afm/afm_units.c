#include <stdlib.h>
#include "mkfn_ext.h"

static int
afm_units(const char *s)
{
	long v;
	int k;

	v = (int)strtol(s, NULL, 10);
	k = 72000 / mkfn_res;
	return 10 * (v + k / 20) / k;
}
