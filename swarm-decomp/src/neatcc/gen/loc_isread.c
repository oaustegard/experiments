#include "ic_ent.h"
static int loc_isread(int reg)
{
	int i = ic_i + 1;
	long k;
	struct ic_ent *p = ic;

	for (k = i; k < ic_n; k++) {
		if (ic_bbeg[k])
			return 0;
		if (p[k].op & 0x200000)
			return p[k].op & 0x10000;
	}
	return 0;
}
