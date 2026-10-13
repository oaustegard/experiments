#include "tr_args.h"
#include "ext.h"

static void tr_ss(struct tr_args *a)
{
	char *s;
	int *p;
	int v;

	if (a->arg[0]) {
		p = nreg(map(".ss"));
		s = a->arg[0];
		v = *p;
		p = nreg(map(".ss"));
		*p = eval_re(s, v, 0);
		if (a->arg[1]) {
			p = nreg(map(".sss"));
			v = eval_re(a->arg[1], *p, 0);
		} else
			v = *nreg(map(".ss"));
		p = nreg(map(".sss"));
		*p = v;
	}
}
