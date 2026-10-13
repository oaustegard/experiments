#include "wb.h"
#include "fn_sbuf.h"
#include "wb_flushsub.h"

char *wb_buf(struct wb *w)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	return sbuf_buf(&w->buf);
}
