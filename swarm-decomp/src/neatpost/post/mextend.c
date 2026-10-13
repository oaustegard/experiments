#include "sys_std.h"

void *mextend(void *p, long n, long nn, int sz)
{
	void *q;

	q = malloc(sz * nn);
	memcpy(q, p, sz * n);
	memset((char *)q + sz * n, 0, (nn - n) * sz);
	free(p);
	return q;
}
