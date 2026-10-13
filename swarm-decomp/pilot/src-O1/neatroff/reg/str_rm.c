#include <stdlib.h>

void str_rm(int i)
{
	if (sregs[i])
		free(sregs[i]);
	sregs[i] = 0;
	sregs_dat[i] = 0;
}
