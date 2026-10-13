#include <time.h>
#include "env.h"
#include "dev_res.h"
#include "fn_map.h"
#include "fn_nreg.h"

/* tentative definition: env_id.c may be included after this file */
static int env_id;

void env_init(void)
{
	time_t t;
	struct tm *tm;
	int i, *p;

	t = time(0);
	tm = localtime(&t);
	num_set(map("dw"), tm->tm_wday + 1);
	num_set(map("dy"), tm->tm_mday);
	num_set(map("mo"), tm->tm_mon + 1);
	num_set(map("yr"), tm->tm_year % 100);
	num_set(map(".yr"), tm->tm_year + 1900);
	p = nreg(111);
	*p = dev_res;
	*nreg(map(".o0")) = *p;
	*nreg(112) = dev_res * 11;
	*nreg(map(".lg")) = 1;
	*nreg(map(".kn")) = 1;
	num_set(map(".H"), 1);
	num_set(map(".V"), 1);
	for (i = 1; i < 48; i++)
		eregs_idx[map(eregs[i - 1])] = i;
	env_id = map("0");
	env = envs[env_id];
	if (!env)
		env_set(env_id);
}
