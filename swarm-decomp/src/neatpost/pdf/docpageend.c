#include <stdio.h>
#include <string.h>
#include "fn_mextend.h"
#include "fn_sbuf_len.h"
#include "fn_sbuf_buf.h"
#include "fn_sbuf_free.h"
#include "fn_sbuf_printf.h"
#include "obj_map.h"
#include "o_queued.h"
#include "o_iset.h"
#include "page_n.h"
#include "page_id.h"
#include "page_sz.h"
#include "pdf_pages.h"
#include "pdf_pos.h"
#include "pg.h"
#include "pfonts.h"
#include "pfonts_n.h"
#include "obj_off.h"
#include "xobj.h"
#include "xobj_n.h"
#include "ann.h"
#include "ann_n.h"

void docpageend(void)
{
	struct pfont *e;
	char *buf;
	int *pid;
	int id, cid, n, i, j;

	if (o_queued == 1)
		sbuf_printf(pg, ">] TJ\n");
	o_queued = 0;
	sbuf_printf(pg, "ET\n");
	id = obj_map();
	cid = id;
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	pdfout("  /Length %d\n", sbuf_len(pg) - 1);
	pdfout(">>\n");
	pdfout("stream\n");
	n = sbuf_len(pg);
	buf = sbuf_buf(pg);
	fwrite(buf, n, 1, stdout);
	pdf_pos += n;
	pdfout("endstream\n");
	pdfout("endobj\n\n");
	if (page_n == page_sz) {
		page_sz = page_n + 0x400;
		page_id = mextend(page_id, page_n, page_sz, 4);
	}
	pid = &page_id[page_n++];
	id = obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	*pid = id;
	pdfout("<<\n");
	pdfout("  /Type /Page\n");
	pdfout("  /Parent %d 0 R\n", pdf_pages);
	pdfout("  /Resources <<\n");
	pdfout("    /Font <<");
	for (j = 0; j < pfonts_n; j++) {
		if (o_iset[j]) {
			e = pfonts + j;
			if (e->f894)
				pdfout(" /%s %d 0 R", e->name, e->objid);
			else
				pdfout(" /%s.%d %d 0 R", e->name, e->f888, e->objid);
		}
	}
	pdfout(" >>\n");
	if (xobj_n) {
		pdfout("    /XObject <<");
		for (i = 0; i < xobj_n; i++)
			pdfout(" /FO%d %d 0 R", i, xobj[i]);
		pdfout(" >>\n");
	}
	pdfout("  >>\n");
	pdfout("  /Contents %d 0 R\n", cid);
	if (ann_n) {
		pdfout("  /Annots [");
		for (i = 0; i < ann_n; i++)
			pdfout(" %d 0 R", ann[i]);
		pdfout(" ]\n");
	}
	pdfout(">>\n");
	pdfout("endobj\n\n");
	sbuf_free(pg);
	memset(o_iset, 0, pfonts_n);
	xobj_n = 0;
	ann_n = 0;
}
