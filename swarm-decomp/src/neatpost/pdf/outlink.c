#include "fn_sbuf_printf.h"
#include "fn_pdftext_static.h"

void outlink(char *s, int x, int y) {
    int n, id;
    if (ann_n == 0x80)
        return;
    tj_flush();
    n = ann_n++;
    id = obj_map();
    obj_off[id] = pdf_pos;
    pdfout("%d 0 obj\n", id);
    ann[n] = id;
    pdfout("<<\n");
    pdfout("  /Type /Annot\n");
    pdfout("  /Subtype /Link\n");
    pdfout("  /Rect [%s", pdfpos00(o_h * 100, o_v * 100));
    pdfout(" %s]\n", pdfpos00((o_h + x) * 100, (o_v + y) * 100));
    if (*s == '#')
        pdfout("  /A << /S /GoTo /D (%s) >>\n", s + 1);
    else
        pdfout("  /A << /S /URI /URI %s >>\n", pdftext_static(s));
    pdfout(">>\n");
    pdfout("endobj\n\n");
}
