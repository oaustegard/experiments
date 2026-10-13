#include <stdlib.h>
#include <string.h>
#include "fn_xmalloc.h"
#include "env.h"

void str_set(int r, char *s)
{
	int n;

	n = strlen(s) + 1;
	if (sregs[r])
		free(sregs[r]);
	sregs[r] = xmalloc(n);
	memcpy(sregs[r], s, n);
	sregs_dat[r] = 0;
}
