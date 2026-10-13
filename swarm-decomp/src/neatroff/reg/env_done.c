#include <stdlib.h>
#include "env.h"
#include "fn_fmt_free.h"
#include "fn_wb_done.h"

void env_done(void)
{
	int i;
	struct env *e;

	for (i = 0; i < 8192; i++) {
		e = envs[i];
		if (e) {
			fmt_free(e->fmt);
			wb_done(&e->wb);
			free(e);
		}
	}
	for (i = 0; i < 8192; i++)
		free(sregs[i]);
}
