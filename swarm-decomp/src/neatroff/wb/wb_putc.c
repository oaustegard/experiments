#include <stdlib.h>
#include "wb.h"
#include "c_ec.h"
#include "wb_putc.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "fn_clr_get.h"
#include "fn_ren_dcmd.h"
#include "fn_sbuf_printf.h"
#include "fn_wb_hmov.h"
#include "fn_wb_put.h"
#include "fn_wb_vmov.h"

static void wb_putc(struct wb *w, int c, char *s)
{
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	switch (c) {
	case 0:
	case 'C':
		if (*s == '\n')
			w->part = 0;
		else
			wb_put(w, s);
		break;
	case '<':
	case '>':
		w->f20[3] = c == '<';
		wb_flushdir(w);
		break;
	case 'D':
		ren_dcmd(w, s);
		break;
	case 'X':
		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		wb_flushdir(w);
		wb_flushfont(w);
		sbuf_printf(&w->buf, "%cX\x02%s\x02", c_ec, s);
		break;
	case 'f':
		w->f20[0] = strtol(s, 0, 10);
		break;
	case 'h':
		wb_hmov(w, strtol(s, 0, 10));
		break;
	case 'm':
		w->f20[2] = clr_get(s);
		break;
	case 's':
		w->f20[1] = strtol(s, 0, 10);
		break;
	case 'v':
		wb_vmov(w, strtol(s, 0, 10));
		break;
	case 'x': {
		int n = strtol(s, 0, 10);

		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		if (n > w->a38[1])
			w->a38[1] = n;
		if (n < w->a38[0])
			w->a38[0] = n;
		sbuf_printf(&w->buf, "%cx'%du'", c_ec, n);
		break;
	}
	}
}
