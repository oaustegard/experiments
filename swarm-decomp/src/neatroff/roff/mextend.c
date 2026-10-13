#include <stdlib.h>
#include <string.h>
#include "fn_mextend.h"
#include "fn_errdie.h"

void *mextend(void *p, long n, long m, int size)
{
	void *q;

	q = malloc(size * m);
	if (q == NULL)
		errdie("neatroff: malloc() failed\n");
	memcpy(q, p, size * n);
	memset((char *)q + size * n, 0, size * (m - n));
	free(p);
	return q;
}
