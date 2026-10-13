#include "wb.h"
#include "c_ec.h"
#include "var_dev_res.h"
#include "fn_nreg.h"
#include "fn_clr_str.h"
#include "fn_sbuf_printf.h"

static void wb_flushfont(struct wb *w)
{
	int f, s, c, top;

	if (w->f10[0] != (w->f20[0] < 0 ? *nreg('f') : w->f20[0])) {
		f = w->f20[0] < 0 ? *nreg('f') : w->f20[0];
		sbuf_printf(&w->buf, "%cf(%02d", c_ec, f);
		w->f10[0] = w->f20[0] < 0 ? *nreg('f') : w->f20[0];
	}
	if (w->f10[1] != (w->f20[1] < 0 ? *nreg('s') : w->f20[1])) {
		if ((w->f20[1] >= 0 ? w->f20[1] : *nreg('s')) <= 99)
			sbuf_printf(&w->buf, "%cs(%02d", c_ec, w->f20[1] < 0 ? *nreg('s') : w->f20[1]);
		else
			sbuf_printf(&w->buf, "%cs[%d]", c_ec, w->f20[1] < 0 ? *nreg('s') : w->f20[1]);
		w->f10[1] = w->f20[1] < 0 ? *nreg('s') : w->f20[1];
	}
	if (*nreg('C') == 0) {
		if (w->f10[2] != (w->f20[2] < 0 ? *nreg('m') : w->f20[2])) {
			c = w->f20[2] < 0 ? *nreg('m') : w->f20[2];
			sbuf_printf(&w->buf, "%cm[%s]", c_ec, clr_str(c));
			w->f10[2] = w->f20[2] < 0 ? *nreg('m') : w->f20[2];
		}
	}
	top = w->vpos - w->f10[1] * dev_res / 72;
	w->a48[2] = top < w->a48[2] ? top : w->a48[2];
	w->a48[1] = w->a48[1] > w->vpos ? w->a48[1] : w->vpos;
}
