#include "ren_ft.h"
#include "fn_dev_pos.h"
#include "fn_errmsg.h"
#include "fn_map.h"
#include "fn_nreg.h"

static void ren_ft(char *s)
{
	int f;

	if (s == 0 || *s == 0 || strcmp(s, "P") == 0)
		f = *nreg(map(".f0"));
	else
		f = dev_pos(s);
	if (f < 0) {
		errmsg("neatroff: failed to mount <%s>\n", s);
		return;
	}
	*nreg(map(".f0")) = *nreg('f');
	*nreg('f') = f;
}
