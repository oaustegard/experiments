#include <stdio.h>
#include "fn_sbuf_printf.h"
static char *pdfpos00(int x, int y) {
    static char buf[64];
    int X = (long)x * 72 / dev_res;
    int Y = 100 * pdf_height - (int)((long)y * 72 / dev_res);
    int ay = Y < 0 ? -Y : Y;
    sprintf(buf, "%s%d.%02d %s%d.%02d", X < 0 ? "-" : "", (X < 0 ? -X : X) / 100, (X < 0 ? -X : X) % 100,
            Y < 0 ? "-" : "", ay / 100, ay % 100);
    return buf;
}
