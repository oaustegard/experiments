#include "wb.h"
#include "glyph.h"
#include "fn_dev_glyph.h"
#include "fn_dev_font.h"
#include "fn_font_gwid.h"
#include "fn_nreg.h"

int wb_hywid(struct wb *w)
{
	struct glyph *g = dev_glyph("hy", w->f10[0]);
	int size, wid, font;

	if (g != 0) {
		size = w->f20[1];
		wid = g->a48;
		if (size < 0)
			size = *nreg('s');
		font = w->f20[0];
		if (font < 0)
			font = *nreg('f');
		return font_gwid(g->font, dev_font(font), size, wid);
	} else
		return 0;
}
