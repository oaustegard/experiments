#include "ren_cl.h"
#include "fn_clr_get.h"
#include "fn_map.h"
#include "fn_nreg.h"

static void ren_cl(char *s)
{
	int c;

	if (s == 0 || *s == 0)
		c = *nreg(map(".m0"));
	else
		c = clr_get(s);
	*nreg(map(".m0")) = *nreg('m');
	*nreg('m') = c;
}
