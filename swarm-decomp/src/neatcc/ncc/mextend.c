#include "sys.h"

void *mextend(void *p, long oldn, long newn, long sz)
{
	void *q = malloc(newn * sz);
	memcpy(q, p, oldn * sz);
	memset((char *)q + oldn * sz, 0, (newn - oldn) * sz);
	free(p);
	return q;
}
