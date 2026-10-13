#include "ps_globals.h"
#include "outf.h"

void drawc(int x)
{
	if (!draw_point)
		outf("%d %d m ", o_h, o_v);
	o_h += x;
	draw_point = 1;
	outf("%d %d drawe ", x, x);
}
