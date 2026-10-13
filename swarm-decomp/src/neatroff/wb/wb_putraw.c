#include "wb.h"
#include "wb_flushsub.h"
#include "fn_wb_put.h"

void wb_putraw(struct wb *w, char *s)
{
	int saved = w->f206c;

	if (w->f2068 && w->f206c)
		goto flush;
skip:
	w->f206c = 0;
	if (*s == '\n')
		goto part0;
put:
	wb_put(w, s);
	w->f206c = saved;
	return;
flush:
	wb_flushsub(w);
	saved = w->f206c;
	w->f206c = 0;
	if (*s != '\n')
		goto put;
part0:
	w->part = 0;
	w->f206c = saved;
}
