#include "region.h"

int reg_lmap(long pos, long key)
{
	int i;

	for (i = 0; i < rgn_n; i++)
		if (rgn[i].key == key && rgn[i].lo <= pos && pos < rgn[i].hi)
			return rgn[i].reg;
	return -1;
}
