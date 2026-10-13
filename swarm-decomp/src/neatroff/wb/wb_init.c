#include <string.h>
#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_empty.h"
void wb_init(struct wb *w)
{
	memset(w, 0, 0x2070);
	sbuf_init((struct sbuf *)w);
	for (int i = 0; i < 4; i++) {
		w->f10[i] = -1;
		w->f20[i] = -1;
	}
	w->a54[0] = 0x20000000;
	w->a54[1] = 0x20000000;
	w->a54[2] = (int)0xe0000000;
	w->a54[3] = (int)0xe0000000;
	w->f206c = 1;
}
