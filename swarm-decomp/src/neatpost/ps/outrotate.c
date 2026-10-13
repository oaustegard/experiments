#include "outf.h"
#include "ps_state.h"
#include "ps_vars.h"

void outrotate(int deg)
{
	o_flush();
	out_fontup(o_f);
	if (o_rdeg)
		outf("%d %d %d rot\n", -o_rdeg, o_rh, o_rv);
	o_rdeg = deg;
	o_rh = o_h;
	o_rv = o_v;
	outf("%d %d %d rot\n", deg, o_h, o_v);
}
