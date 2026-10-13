#include "ps_globals.h"
#include "outf.h"

void drawend(int close, int fill)
{
	if (draw_path)
		return;
	draw_point = 0;
	if (close) {
		outf("closepath ");
		if (fill)
			outf("fill\n");
		else
			outf("stroke\n");
	} else if (fill)
		outf("fill\n");
	else
		outf("stroke\n");
}
