#include "vars.h"
#include "fn_in_ex.h"
#include "in_pop.h"

void
in_ex(void)
{
	if (buf == NULL) {
		cfile = nfiles;
		return;
	}
	do
		in_pop();
	while (buf != NULL);
	cfile = nfiles;
}
