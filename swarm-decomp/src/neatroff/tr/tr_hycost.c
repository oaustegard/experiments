#include "tr_args.h"
#include "fn_eval_re.h"
#include "fn_map.h"
#include "fn_nreg.h"

static void tr_hycost(struct tr_args *a)
{
	int n = 0;

	if (a->arg[0])
		n = eval_re(a->arg[0], *nreg(map(".hycost")), 0);
	*nreg(map(".hycost")) = n;
	n = 0;
	if (a->arg[1])
		n = eval_re(a->arg[1], *nreg(map(".hycost2")), 0);
	*nreg(map(".hycost2")) = n;
	n = 0;
	if (a->arg[2])
		n = eval_re(a->arg[2], *nreg(map(".hycost3")), 0);
	*nreg(map(".hycost3")) = n;
}
