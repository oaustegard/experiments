#include "outf.h"
#include "ps_state.h"

static void o_flush(void)
{
	if (o_qtype == 1)
		outf(") %d %d w\n", o_qh, o_qv);
	if (o_qtype == 2)
		outf("] %d %d g\n", o_qh, o_qv);
	o_qtype = 0;
}
