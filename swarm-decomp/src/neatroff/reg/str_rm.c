#include <stdlib.h>

void str_rm(int r)
{
	if (sregs[r])
		free(sregs[r]);
	sregs[r] = 0;
	sregs_dat[r] = 0;
}
