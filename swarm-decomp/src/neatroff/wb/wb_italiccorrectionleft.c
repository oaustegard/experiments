#include "wb.h"
#include "wb_flushsub.h"

void wb_italiccorrectionleft(struct wb *w)
{
	if (!w->f2068 || !w->f206c) {
		*(int *)w->pad64 = 1;
		return;
	}
	wb_flushsub(w);
	*(int *)w->pad64 = 1;
}
