#include <stdlib.h>
#include "fn_errdie.h"
#include "fn_xmalloc.h"

void *xmalloc(size_t n)
{
	void *p = malloc(n);
	if (p == NULL)
		errdie("neatroff: malloc() failed\n");
	return p;
}
