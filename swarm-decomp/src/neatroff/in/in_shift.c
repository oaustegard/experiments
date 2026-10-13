#include "vars.h"
#include "fn_in_shift.h"
#include "fn_free.h"
#include "fn_memset.h"

void
in_shift(void)
{
	struct src *p = buf;

	while (p && p->args == NULL)
		p = p->prev;
	if (p == NULL || p->args == NULL)
		return;
	free(p->args[1]);
	memmove(&p->args[1], &p->args[2], 30 * sizeof(char *));
	p->args[31] = NULL;
}
