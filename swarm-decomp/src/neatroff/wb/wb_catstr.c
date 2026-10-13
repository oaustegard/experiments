#include "wb.h"
#include "wb_flushsub.h"
#include "wb_putc.h"
#include "fn_escread.h"

void wb_catstr(struct wb *w, char *s, char *end)
{
	int flag = w->f206c;
	char *x;
	int c;

	if (w->f2068 && flag) {
		wb_flushsub(w);
		flag = w->f206c;
	}
	w->f206c = 0;
	while (s < end) {
		c = escread(&s, &x);
		if (c < 0)
			break;
		wb_putc(w, c, x);
	}
	w->f206c = flag;
}
