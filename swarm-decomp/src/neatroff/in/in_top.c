#include "vars.h"
#include "fn_in_top.h"

int
in_top(void)
{
	struct src *p = buf;
	int n;

	if (p == NULL)
		return -1;
	n = p->nback;
	if (n == 0)
		return -1;
	return p->back[n - 1];
}
