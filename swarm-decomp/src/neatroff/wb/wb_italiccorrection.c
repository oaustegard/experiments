#include "wb.h"
#include "c_ec.h"
#include "glyph.h"
#include "fn_dev_glyph.h"
#include "fn_font_wid.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

/* font_wid is declared unprototyped in types/; this call uses its real 3-arg type */
#define font_wid3(f, n, k) ((int (*)(void *, int, int))font_wid)(f, n, k)
void wb_italiccorrection(struct wb *w)
{
	struct glyph *g;
	int wid, d;

	if (w->f2068 <= 0)
		return;
	g = dev_glyph((char *)w + 0x68 + (long)(w->f2068 - 1) * 32, w->f10[0]);
	if (g == 0)
		return;
	d = g->a4e - g->a48;
	if (g->a4e != g->a48) {
		if (d > 0) {
			wid = font_wid3(g->font, w->f10[1], d);
			wb_flushsub(w);
			w->hpos += wid;
			sbuf_printf(&w->buf, "%ch'%du'", c_ec, wid);
		}
	}
}
