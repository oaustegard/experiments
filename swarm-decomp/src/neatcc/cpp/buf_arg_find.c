#include "cpp_fns.h"

static int buf_arg_find(char *name)
{
	int i;
	unsigned n;
	struct macro *m;

	for (i = bufs_n - 1; i >= 0; i--) {
		if (bufs[i].kind == 1) {
			m = bufs[i].macro;
			for (n = 0; n < m->nargs; n++)
				if (!strcmp(name, m->arg[n]))
					return i;
		} else if (bufs[i].kind == 2)
			i = bufs[i].argi;
	}
	return -1;
}
