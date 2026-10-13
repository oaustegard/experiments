#include "wb.h"
#include "c_ec.h"
#include "var_dir_do.h"
#include "wb_flushsub.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_sbuf_printf.h"

void wb_flushdir(struct wb *w)
{
	int cur, cur0, d;
	char c;

	cur0 = w->f10[3];
	cur = w->f20[3];
	if (cur < 0)
		cur = *nreg(map(".cd"));
	if (cur0 == cur)
		return;
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	if (dir_do) {
		d = w->f20[3];
		c = '>';
		if (d < 0) {
			if (*nreg(map(".cd")) > 0)
				c = '<';
		} else if (d != 0) {
			c = '<';
		}
		sbuf_printf(&w->buf, "%c%c", c_ec, c);
	}
	w->f10[3] = w->f20[3] < 0 ? *nreg(map(".cd")) : w->f20[3];
}
