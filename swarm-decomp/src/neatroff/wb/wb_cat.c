#include "wb.h"
#include "wb_flushsub.h"
#include "wb_putc.h"
#include "fn_escread.h"
#include "fn_sbuf.h"
#include "fn_wb_reset.h"

void wb_cat(struct wb *w, struct wb *v)
{
	char *s, *x;
	int c, flag, part, i;

	if (v->f2068 && v->f206c)
		wb_flushsub(v);
	flag = w->f206c;
	if (w->f2068 && flag) {
		wb_flushsub(w);
		flag = w->f206c;
	}
	w->f206c = 0;
	s = sbuf_buf(&v->buf);
	while ((c = escread(&s, &x)) >= 0)
		wb_putc(w, c, x);
	part = v->part;
	for (i = 0; i < 4; i++)
		w->f20[i] = -1;
	wb_reset(v);
	v->part = part;
	w->f206c = flag;
}
