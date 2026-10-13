#include "env.h"
#include "fn_map.h"
#include "tr_args.h"
#include "env_set.h"

void tr_ev(struct tr_args *a)
{
	int id;

	if (a->arg[0]) {
		id = map(a->arg[0]);
		if (id < 0)
			return;
	} else {
		if (nenv == 0)
			return;
		nenv--;
		id = oenv[nenv];
		if (id < 0)
			return;
	}
	if (a->arg[0] && env && nenv < 16) {
		oenv[nenv++] = env_id;
	}
	env_id = id;
	env = envs[id];
	if (!env)
		env_set(id);
}
