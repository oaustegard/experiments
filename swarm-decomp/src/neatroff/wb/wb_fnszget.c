#include "wb.h"
#include "wb_flushsub.h"

void wb_fnszget(struct wb *w, int *a, int *b, int *c, int *d)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	*a = w->f20[0];
	*b = w->f20[1];
	*c = w->f20[2];
	*d = w->f20[3];
}
