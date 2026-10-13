#include "vars.h"
#include "fn_free.h"
#include "libc.h"
#include "in_pop.h"

static void
in_pop(void)
{
	struct src *p = buf;
	char **a;
	int i;

	buf = p->prev;
	a = p->args;
	if (a != NULL) {
		for (i = 0; i < 32; i++)
			if (a[i])
				free(a[i]);
		free(a);
	}
	if (p->fp != NULL && p->fp != stdin)
		fclose(p->fp);
	free(p->name);
	free(p);
}
