#include <string.h>
#include "env.h"
#include "envs.h"
#include "env_id.h"
#include "fn_map.h"
#include "fn_xmalloc.h"
#include "fn_fmt_alloc.h"
#include "fn_fmt_free.h"
#include "fn_wb_init.h"
#include "fn_wb_done.h"
#include "fn_free.h"
#include "tr_args.h"

void tr_evc(struct tr_args *a)
{
	struct env *o, *n, *old;
	int id;

	if (!a->arg[0])
		return;
	id = map(a->arg[0]);
	if (id < 0)
		return;
	o = envs[id];
	if (!o)
		return;
	n = xmalloc(sizeof *n);
	memset(n, 0, sizeof *n);
	wb_init(&n->wb);
	n->fmt = fmt_alloc();
	env = n;
	memcpy(n->reg, o->reg, sizeof n->reg);
	memcpy(n->tab, o->tab, sizeof n->tab);
	memcpy(n->pad180, o->pad180, sizeof n->pad180);
	memcpy(n->tc, o->tc, sizeof n->tc + sizeof n->lc + sizeof n->hc + sizeof n->mc);
	old = envs[env_id];
	fmt_free(old->fmt);
	wb_done(&old->wb);
	free(old);
	envs[env_id] = env;
}
