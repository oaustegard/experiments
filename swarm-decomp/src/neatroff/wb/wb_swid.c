#include "wb.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_dev_font.h"
#include "fn_font_swid.h"

int wb_swid(struct wb *w)
{
	int ss = *nreg(map(".ss"));
	int sz = w->f20[1];
	int fn;

	if (sz < 0)
		sz = *nreg('s');
	fn = w->f20[0];
	if (fn < 0)
		fn = *nreg('f');
	return font_swid(dev_font(fn), sz, ss);
}
