#include <string.h>
#include "env.h"
#include "fn_map.h"
#include "fn_memset.h"
#include "fn_nreg.h"
#include "fn_xmalloc.h"
#include "fn_wb_init.h"
#include "fn_fmt_alloc.h"
#include "dev_res.h"

/* tentative definition: envs.c sorts after this file */
static struct env *envs[8192];

static void env_set(int i)
{
	struct env *e;
	int *p, *q;
	int j, t;

	if (envs[i]) {
		env = envs[i];
		return;
	}
	e = xmalloc(sizeof(struct env));
	memset(e, 0, sizeof(struct env));
	wb_init(&e->wb);
	e->fmt = fmt_alloc();
	envs[i] = e;
	env = e;
	*nreg(102) = 1;
	*nreg(105) = 0;
	*nreg(73) = 0;
	*nreg(106) = 16;
	*nreg(108) = dev_res * 65 / 10;
	*nreg(76) = 1;
	p = nreg(115);
	*p = 10;
	*nreg(117) = 1;
	*nreg(118) = dev_res / 72 * 12;
	*nreg(map(".s0")) = *p;
	q = nreg(118);
	*nreg(map(".v0")) = *q;
	q = nreg(102);
	*nreg(map(".f0")) = *q;
	q = nreg(108);
	*nreg(map(".l0")) = *q;
	q = nreg(76);
	*nreg(map(".L0")) = *q;
	*nreg(map(".na")) = 0;
	t = dev_res * 65;
	*nreg(map(".lt")) = t / 10;
	*nreg(map(".hy")) = 1;
	*nreg(map(".ss")) = 12;
	*nreg(map(".sss")) = 12;
	*nreg(map(".nM")) = 1;
	*nreg(map(".nS")) = 1;
	env->hc[0] = '\\';
	env->hc[1] = '%';
	env->hc[2] = 0;
	env->lc[0] = '.';
	env->lc[1] = 0;
	for (j = 0; j < 32; j++)
		env->tab[j] = j * dev_res / 2;
	memset(env->pad180, 'L', 32);
}
