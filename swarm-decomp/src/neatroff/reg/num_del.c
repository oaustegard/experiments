#include "env.h"

void num_del(int r)
{
	*nreg(r) = 0;
	nregs_inc[r] = 0;
	nregs_fmt[r] = 0;
}
