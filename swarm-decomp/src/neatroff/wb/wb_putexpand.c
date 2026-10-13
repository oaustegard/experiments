#include "wb.h"
#include "fn_cdef_expand.h"
#include "fn_nreg.h"
#include "fn_wb_put.h"

void wb_putexpand(struct wb *w, char *s)
{
	int f = w->f20[0];

	if (f < 0)
		f = *nreg('f');
	if (!cdef_expand((char *)w, s, f))
		return;
	if (*s == '\n') {
		w->part = 0;
		return;
	}
	wb_put(w, s);
}
