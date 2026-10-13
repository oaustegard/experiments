#include "wb.h"
#include "wb_flushsub.h"

int wb_vpos(struct wb *w)
{
	wb_flushsub(w);
	return w->vpos;
}
