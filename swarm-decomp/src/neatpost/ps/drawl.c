#include "outf.h"
#include "ps_globals.h"
#include "ps_vars.h"

void drawl(int x, int y)
{
	if (!draw_point)
		outf("%d %d m ", o_h, o_v);
	o_h += x;
	draw_point = 1;
	o_v += y;
	outf("%d %d drawl ", o_h, o_v);
}
