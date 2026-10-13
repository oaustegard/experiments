#include "fn_sbuf_printf.h"
#include "dev_res.h"
void draws(int a, int b, int c, int d) {
    int x0 = o_h + a;
    int y0 = o_v + b;
    int y5 = 5 * y0;
    int yd = y0 + d;
    int xc = x0 + c;
    int x5 = 5 * x0;
    sbuf_printf(pg, "%s ", pdfpos00((o_h + x5) / 6 * 100, (o_v + y5) / 6 * 100));
    sbuf_printf(pg, "%s ", pdfpos00((x5 + xc) / 6 * 100, (y5 + yd) / 6 * 100));
    sbuf_printf(pg, "%s c\n", pdfpos00((x0 + xc) / 2 * 100, (y0 + yd) / 2 * 100));
    o_h += a;
    o_v += b;
}
