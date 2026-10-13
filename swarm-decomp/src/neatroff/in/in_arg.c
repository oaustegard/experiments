#include "vars.h"
#include "fn_in_arg.h"

char *
in_arg(int i)
{
	struct src *p = buf;
	char **a;

	while (p && p->args == NULL)
		p = p->prev;
	if (p == NULL)
		return "";
	a = p->args;
	if (i > 31 || a == NULL)
		return "";
	return a[i] ? a[i] : "";
}
