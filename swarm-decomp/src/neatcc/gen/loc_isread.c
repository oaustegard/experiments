#include "ic_ent.h"
static int loc_isread(int reg)
{
	int i;

	for (i = ic_i + 1; i < ic_n; i++) {
		if (ic_bbeg[i])
			return 0;
		if (ic[i].op & 0x200000)
			return ic[i].op & 0x10000;
	}
	return 0;
}
