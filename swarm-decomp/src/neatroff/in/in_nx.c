#include "vars.h"
#include "fn_in_nx.h"
#include "fn_in_so.h"
#include "in_pop.h"

void
in_nx(char *s)
{
	while (buf != NULL)
		in_pop();
	if (s != NULL)
		in_so(s);
}
