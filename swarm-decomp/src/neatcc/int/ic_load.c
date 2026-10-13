#include "ic_ent.h"

static int ic_load(long i)
{
	int j;

	if (!(ic[i].op & 0x10000) || !(ic[i].op & 0x700000))
		return 0;
	for (j = i + 1; j < ic_n; j++)
		if (ic[j].op & 0x20000)
			return 0;
	return 1;
}
