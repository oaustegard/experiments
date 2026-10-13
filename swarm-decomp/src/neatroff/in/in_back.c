#include "vars.h"
#include "fn_in_back.h"

void
in_back(int c)
{
	struct src *p;
	int n;

	if (c < 0)
		return;
	p = buf;
	if (p == NULL)
		return;
	n = p->nback;
	if ((unsigned)n > 127)
		return;
	p->nback = n + 1;
	p->back[n] = c;
}
