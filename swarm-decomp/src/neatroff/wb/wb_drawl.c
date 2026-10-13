#include "wb.h"
#include "c_ec.h"
#include "var_dev_res.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "fn_sbuf_printf.h"

void wb_drawl(struct wb *w, int a, int b, int c)
{
	int top;

	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	wb_flushdir(w);
	wb_flushfont(w);
	sbuf_printf(&w->buf, "%cD'%c %du %du'", c_ec, a, b, c);
	w->hpos += b;
	w->vpos += c;
	top = w->vpos - w->f10[1] * dev_res / 72;
	w->a48[2] = top < w->a48[2] ? top : w->a48[2];
	w->a48[1] = w->a48[1] > w->vpos ? w->a48[1] : w->vpos;
}
