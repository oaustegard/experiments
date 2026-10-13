#include "env.h"

void num_inc(int r, int v)
{
	int i;
	int x;
	int *p;
	int n;

	i = eregs_idx[r];
	if (!i) {
		x = nregs[r];
		p = nregs + r;
	} else {
		p = env->reg + i;
		x = env->reg[i];
	}
	n = nregs_inc[r];
	if (v <= 0)
		n = -n;
	*p = x + n;
}
