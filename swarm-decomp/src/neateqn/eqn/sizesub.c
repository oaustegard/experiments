#include "libc.h"
#include "fn_nreg.h"
#include "fn_nreg.h"
#include "e_vars.h"

static void sizesub(int reg, int ps, int size, int cur)
{
	size >>= 4;
	cur >>= 4;
	if (size > cur) {
		printf(".nr %s %s*7/10\n", nregname(reg), nreg(ps));
		printf(".if %s<%d .nr %s %d\n", nreg(reg), e_minimumsize, nregname(reg), e_minimumsize);
	} else {
		printf(".nr %s %s\n", nregname(reg), nreg(ps));
	}
}
