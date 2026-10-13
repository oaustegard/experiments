#include "wb.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_empty.h"
void wb_done(struct wb *w) { sbuf_done((struct sbuf *)w); }
