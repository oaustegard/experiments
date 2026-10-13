#include "wb.h"
#include "c_ec.h"
#include "fn_sbuf_printf.h"
#include "wb_flushsub.h"

void wb_hmov(struct wb *w, int n)
{
	wb_flushsub(w);
	w->hpos += n;
	sbuf_printf(&w->buf, "%ch'%du'", c_ec, n);
}
