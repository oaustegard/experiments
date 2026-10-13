#include "wb.h"
#include "c_ec.h"
#include "fn_sbuf_printf.h"
#include "wb_flushsub.h"

void wb_vmov(struct wb *w, int n)
{
	wb_flushsub(w);
	w->vpos += n;
	sbuf_printf(&w->buf, "%cv'%du'", c_ec, n);
}
