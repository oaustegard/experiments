#include "outf.h"
#include "ps_globals.h"
#include "ps_vars.h"

void draws(int a, int b, int c, int d)
{
	if (!draw_point)
		outf("%d %d m ", o_h, o_v);
	draw_point = 1;
	outf("%d %d %d %d %d %d draws ", o_h, o_v, o_h + a, o_v + b, o_h + a + c, o_v + b + d);
	o_h += a;
	o_v += b;
}
