#include "wb.h"
#include "c_ec.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_drawc(struct wb *w, int c, int d)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	wb_flushdir(w);
	wb_flushfont(w);
	sbuf_printf(&w->buf, "%cD'%c %du'", c_ec, c, d);
	w->hpos += d;
}
