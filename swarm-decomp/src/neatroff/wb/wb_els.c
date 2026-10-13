#include "wb.h"
#include "c_ec.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_els(struct wb *w, int n)
{
	wb_flushsub(w);
	if (w->a38[1] < n)
		w->a38[1] = n;
	if (w->a38[0] > n)
		w->a38[0] = n;
	sbuf_printf(&w->buf, "%cx'%du'", c_ec, n);
}
