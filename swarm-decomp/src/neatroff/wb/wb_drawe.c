#include "wb.h"
#include "c_ec.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_drawe(struct wb *w, int a, int b, int c)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	wb_flushdir(w);
	wb_flushfont(w);
	sbuf_printf(&w->buf, "%cD'%c %du %du'", c_ec, a, b, c);
	w->hpos += b;
}
