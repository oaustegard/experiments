#include "wb.h"
#include "wb_flushsub.h"

int wb_vpos(struct wb *w)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	return w->vpos;
}
