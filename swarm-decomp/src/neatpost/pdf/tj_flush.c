#include "fn_sbuf_printf.h"
static void tj_flush(void) {
    if (o_queued == 1)
        sbuf_printf(pg, ">] TJ\n");
    o_queued = 0;
}
