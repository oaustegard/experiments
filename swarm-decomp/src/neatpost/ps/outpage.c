#include "outf.h"
#include "ps_state.h"
#include "ps_vars.h"

void outpage(void)
{
	o_flush();
	o_v = 0;
	o_h = 0;
	p_s = 0;
	p_f = 0;
	p_m = 0;
	o_rdeg = 0;
}
