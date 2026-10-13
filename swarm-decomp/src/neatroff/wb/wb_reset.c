#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include <string.h>

void wb_reset(struct wb *w)
{
	int i;

	sbuf_done(&w->buf);
	memset(w, 0, 0x2070);
	sbuf_init(&w->buf);
	w->f206c = 1;
	for (i = 0; i < 8; i++)
		w->a10[i] = -1;
	w->a54[0] = 0x20000000;
	w->a54[1] = 0x20000000;
	w->a54[2] = 0xe0000000;
	w->a54[3] = 0xe0000000;
}
