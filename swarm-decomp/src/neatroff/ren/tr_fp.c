#include <ctype.h>

void tr_fp(struct tr_args *a)
{
	int n = -1;

	if (a->arg[1]) {
		if (isdigit((unsigned char)*a->arg[0]))
			n = strtol(a->arg[0], 0, 10);
		if (dev_mnt(n, a->arg[1], a->arg[2] ? a->arg[2] : a->arg[1]) < 0)
			errmsg("neatroff: failed to mount <%s>\n", a->arg[1]);
	}
}
