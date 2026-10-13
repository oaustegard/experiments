#include "vars.h"
#include "fn_in_nargs.h"

int
in_nargs(void)
{
	struct src *p = buf;
	char **a;
	int n = 0, i;

	while (p && p->args == NULL)
		p = p->prev;
	if (p == NULL)
		return n;
	a = p->args;
	i = 0;
	while (a != NULL && a[++i - 1])
		n = i;
	return n;
}
