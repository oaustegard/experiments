#include "vars.h"
#include "fn_in_nargs.h"

int
in_nargs(void)
{
	struct src *p = buf;
	char **a;
	int n = 0, i;

	if (p == NULL)
		return 0;
	while (p->args == NULL) {
		p = p->prev;
		if (p == NULL)
			return 0;
	}
	a = p->args;
	if (a == NULL)
		return 0;
	for (i = 1; a[i - 1]; i++)
		n = i;
	return n;
}
