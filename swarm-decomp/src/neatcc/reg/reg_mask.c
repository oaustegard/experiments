#include "region.h"

long reg_mask(void)
{
	long m = 0;
	int i;

	for (i = 0; i < rgn_n; i++)
		if (rgn[i].reg >= 0)
			m |= 1 << rgn[i].reg;
	return m;
}
