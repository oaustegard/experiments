#include "region.h"

int reg_rmap(long pos, long reg)
{
	int i;

	for (i = 0; i < rgn_n; i++)
		if (rgn[i].reg == reg && rgn[i].lo <= pos && pos < rgn[i].hi)
			return rgn[i].key;
	return -1;
}
