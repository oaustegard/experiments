#include "fn_sbuf_printf.h"
#include "dev_res.h"
void outname(int n, char (*names)[64], int *pages, int *ys) {
    int i, id;
    long k;
    if (o_queued == 1)
        sbuf_printf(pg, ">] TJ\n");
    o_queued = 0;
    id = obj_map();
    obj_off[id] = pdf_pos;
    pdfout("%d 0 obj\n", id);
    pdf_dests = id;
    pdfout("<<\n");
    for (i = 0; i < n; i++) {
        k = pages[i];
        if (k > 0 && k <= page_n) {
            pdfout("  /%s [ %d 0 R /XYZ 0 %d 0 ]\n", names[i], page_id[k - 1],
                   pdf_height - 72 * ys[i] / dev_res);
        }
    }
    pdfout(">>\n");
    pdfout("endobj\n\n");
}
