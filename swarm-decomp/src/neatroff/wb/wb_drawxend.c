#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_empty.h"
void wb_drawxend(struct wb *w) { sbuf_printf((struct sbuf *)w, "'"); }
