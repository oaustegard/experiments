#include "wb.h"
#include "c_ec.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_etc(struct wb *w, char *s)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	wb_flushdir(w);
	wb_flushfont(w);
	sbuf_printf(&w->buf, "%cX\x02%s\x02", c_ec, s);
}
