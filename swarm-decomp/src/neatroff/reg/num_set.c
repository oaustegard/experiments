#include "env.h"
#include "fn_nreg.h"

void num_set(int r, int v)
{
	if (!nregs_fmt[r])
		nregs_fmt[r] = '0';
	*nreg(r) = v;
}
