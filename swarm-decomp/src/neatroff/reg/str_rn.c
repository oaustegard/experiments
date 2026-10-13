#include <stdlib.h>

void str_rn(int a, int b)
{
	if (sregs[a] || sregs_dat[a]) {
		if (sregs[b])
			free(sregs[b]);
		sregs[b] = 0;
		sregs_dat[b] = 0;
		sregs[b] = sregs[a];
		sregs[a] = 0;
		sregs_dat[b] = sregs_dat[a];
		sregs_dat[a] = 0;
	}
}
