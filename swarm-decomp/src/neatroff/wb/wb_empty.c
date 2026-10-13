#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_empty.h"
int wb_empty(struct wb *w)
{
	if (w->f2068)
		return 0;
	return sbuf_empty((struct sbuf *)w) != 0;
}
