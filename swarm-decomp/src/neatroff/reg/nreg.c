#include "env.h"

int *nreg(int r)
{
	int i = eregs_idx[r];

	if (i)
		return (int *)env + i;
	return nregs + r;
}
