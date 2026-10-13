#include "fn_sbuf_printf.h"
#include "pfont.h"
#include "pdfcolor.h"
static void out_fontup(void) {
    struct pfont *f;
    if (o_m != p_m) {
        tj_flush();
        sbuf_printf(pg, "%s rg\n", pdfcolor(o_m));
        p_m = o_m;
    }
    if (o_i >= 0 && (o_i != p_i || o_s != p_s)) {
        f = pfonts + o_i;
        tj_flush();
        if (f->f894 != 0)
            sbuf_printf(pg, "/%s %d Tf\n", f->name, o_s);
        else
            sbuf_printf(pg, "/%s.%d %d Tf\n", f->name, f->f888, o_s);
        p_i = o_i;
        p_s = o_s;
    }
}
