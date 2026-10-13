#include "tr_args.h"
#include "ext.h"
#include "libc.h"

static void tr_nm(struct tr_args *a)
{
	char *s;

	if (!a->arg[0]) {
		*nreg(map(".nm")) = 0;
		return;
	}
	*nreg(map(".nm")) = 1;
	s = a->arg[0];
	if (strchr("0123456789+-(", *s) && strpbrk(s, "0123456789")) {
		int v = 0;
		if (eval_re(a->arg[0], *nreg(map("ln")), 0) > 0)
			v = eval_re(a->arg[0], *nreg(map("ln")), 0);
		*nreg(map("ln")) = v;
	}
	s = a->arg[1];
	if (s && strchr("0123456789+-(", *s) && strpbrk(s, "0123456789")) {
		int v = 1;
		if (eval(s, 0) > 1)
			v = eval(a->arg[1], 0);
		*nreg(map(".nM")) = v;
	}
	s = a->arg[2];
	if (s && strchr("0123456789+-(", *s) && strpbrk(s, "0123456789")) {
		int v = 0;
		if (eval(s, 0) > 0)
			v = eval(a->arg[2], 0);
		*nreg(map(".nS")) = v;
	}
	s = a->arg[3];
	if (s && strchr("0123456789+-(", *s) && strpbrk(s, "0123456789")) {
		int v = 0;
		if (eval(s, 0) > 0)
			v = eval(a->arg[3], 0);
		*nreg(map(".nI")) = v;
	}
}
