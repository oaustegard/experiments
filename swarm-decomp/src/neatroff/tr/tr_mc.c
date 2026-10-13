#include "tr_args.h"
#include "ext.h"
#include "fn_eval.h"
#include "var_dev_res.h"

static void tr_mc(struct tr_args *a)
{
	char *s = a->arg[0];
	int v;

	if (s && charread(&s, env_mc()) >= 0) {
		*nreg(map(".mc")) = 1;
		if (a->arg[1])
			v = eval(a->arg[1], 'm');
		else
			v = *nreg('s') * dev_res / 72;
		*nreg(map(".mcn")) = v;
		return;
	}
	*nreg(map(".mc")) = 0;
}
