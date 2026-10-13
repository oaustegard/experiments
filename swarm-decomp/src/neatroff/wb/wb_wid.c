#include "wb.h"
#include "wb_flushsub.h"

int wb_wid(struct wb *w)
{
	wb_flushsub(w);
	return w->hpos;
}
