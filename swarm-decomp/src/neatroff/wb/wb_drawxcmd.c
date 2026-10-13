#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_empty.h"
void wb_drawxcmd(struct wb *w, const char *s) { sbuf_printf((struct sbuf *)w, " %s", s); }
