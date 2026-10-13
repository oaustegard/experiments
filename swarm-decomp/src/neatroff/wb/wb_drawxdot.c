#include "wb.h"
#include "var_dev_res.h"
#include "fn_sbuf_printf.h"

void wb_drawxdot(struct wb *w, int x, int y)
{
	int top;

	sbuf_printf(&w->buf, " %du %du", x, y);
	w->hpos += x;
	w->vpos += y;
	top = w->vpos - w->f10[1] * dev_res / 72;
	w->a48[2] = top < w->a48[2] ? top : w->a48[2];
	w->a48[1] = w->a48[1] > w->vpos ? w->a48[1] : w->vpos;
}
