#include <string.h>
#include "func.h"
#include "ncc_data.h"
#include "decl.h"
#include "krdef.h"

/* a K&R parameter declaration: copies its type into the parameter of the
   function (index local) whose name matches */
static void krdef(long local, struct decl *d, int storage)
{
	struct func *f = (struct func *)funcs + local;
	int i;

	for (i = 0; i < f->nargs; i++)
		if (strcmp(f->arg[i], d->name) == 0)
			memcpy(&f->param[i], &d->t, sizeof(struct type));
}
