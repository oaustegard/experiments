#include "wb.h"
#include "wb_flushsub.h"
#include "fn_wb_put.h"

void wb_putraw(struct wb *w, char *s)
{
	int saved = w->f206c;

	if (w->f2068 && w->f206c) {
		wb_flushsub(w);
		saved = w->f206c;
	}
	w->f206c = 0;
	if (*s == '\n')
		w->part = 0;
	else
		wb_put(w, s);
	w->f206c = saved;
}
