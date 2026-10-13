#include "wb.h"
#include "c_ec.h"
#include "glyph.h"
#include "fn_dev_glyph.h"
#include "fn_font_wid.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_italiccorrection(struct wb *w)
{
	struct glyph *g;
	int wid, d;

	if (w->f2068) {
		g = dev_glyph((char *)w + 0x68 + (long)(w->f2068 - 1) * 32, w->f10[0]);
		if (g) {
			d = g->a4e - g->a48;
			if (d > 0 && g->a4e != g->a48) {
				wid = font_wid(g->font, w->f10[1]);
				if (w->f2068 && w->f206c)
					wb_flushsub(w);
				w->hpos += wid;
				sbuf_printf(&w->buf, "%ch'%du'", c_ec, wid);
			}
		}
	}
}
