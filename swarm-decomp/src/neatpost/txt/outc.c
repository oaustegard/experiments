#include "fn_dev_glyph.h"

void outc(const char *s)
{
	struct fglyph *g;
	int x, y;

	g = dev_glyph(s, o_f);
	y = o_v / c_ht - 1;
	x = o_h / c_wd;
	if (x >= 0 && y >= 0 && x < p_cwd && y < p_cht)
		((unsigned char *)o_pg)[p_cwd * y + x] = ((unsigned char *)g)[0x20];
}
