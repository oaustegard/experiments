#include <stdlib.h>
#include <string.h>
#include "fn_dev_fontopen.h"
#include "fn_font_glget.h"
#include "fn_font_close.h"
#include "fn_pdftext_static.h"
#include "dev_res.h"
#include "fonttype.h"
#include "obj_map.h"

void doctrailer(void)
{
	struct pfont *e;
	struct font *f;
	struct fglyph *g;
	int i, k, id, cid, xref;

	id = pdf_pages > 0 ? pdf_pages : obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	pdfout("  /Type /Pages\n");
	pdfout("  /MediaBox [ 0 0 %d %d ]\n", pdf_width, pdf_height);
	pdfout("  /Count %d\n", page_n);
	pdfout("  /Kids [");
	for (i = 0; i < page_n; i++)
		pdfout(" %d 0 R", page_id[i]);
	pdfout(" ]\n");
	pdfout(">>\n");
	pdfout("endobj\n\n");

	id = pdf_root > 0 ? pdf_root : obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	pdfout("  /Type /Catalog\n");
	pdfout("  /Pages %d 0 R\n", pdf_pages);
	if (pdf_dests > 0)
		pdfout("  /Dests %d 0 R\n", pdf_dests);
	if (pdf_outline > 0)
		pdfout("  /Outlines %d 0 R\n", pdf_outline);
	pdfout(">>\n");
	pdfout("endobj\n\n");

	for (i = 0; i < pfonts_n; i++) {
		e = pfonts + i;
		if (e->f894) {
			f = dev_fontopen(e->desc);
			cid = obj_map();
			obj_off[cid] = pdf_pos;
			pdfout("%d 0 obj\n", cid);
			pdfout("<<\n");
			pdfout("  /Type /Font\n");
			pdfout("  /Subtype /CIDFontType2\n");
			pdfout("  /BaseFont /%s\n", e->name);
			pdfout("  /CIDSystemInfo <</Ordering(Identity)/Registry(Adobe)/Supplement 0>>\n");
			pdfout("  /FontDescriptor %d 0 R\n", e->wid);
			pdfout("  /DW 1000\n");
			for (k = 0; font_glget(f, k); k++)
				;
			pdfout("  /W [ %d [", e->minc);
			for (k = e->minc; k <= e->maxc; k++) {
				g = font_glget(f, k);
				pdfout(" %d", (long)g->width * 0x1c20 / dev_res);
			}
			pdfout(" ] ]\n");
			pdfout(">>\n");
			pdfout("endobj\n\n");
			id = e->objid > 0 ? e->objid : obj_map();
			obj_off[id] = pdf_pos;
			pdfout("%d 0 obj\n", id);
			pdfout("<<\n");
			pdfout("  /Type /Font\n");
			pdfout("  /Subtype /Type0\n");
			pdfout("  /BaseFont /%s\n", e->name);
			pdfout("  /Encoding /Identity-H\n");
			pdfout("  /DescendantFonts [%d 0 R]\n", cid);
			pdfout(">>\n");
			pdfout("endobj\n\n");
			font_close(f);
		} else {
			f = dev_fontopen(e->desc);
			cid = obj_map();
			obj_off[cid] = pdf_pos;
			pdfout("%d 0 obj\n", cid);
			pdfout("<<\n");
			pdfout("  /Type /Encoding\n");
			pdfout("  /Differences [ %d", e->minc % 256);
			for (k = e->minc; k <= e->maxc; k++)
				pdfout(" /%s", font_glget(f, k)->name);
			pdfout(" ]\n");
			pdfout(">>\n");
			pdfout("endobj\n\n");
			id = e->objid > 0 ? e->objid : obj_map();
			obj_off[id] = pdf_pos;
			pdfout("%d 0 obj\n", id);
			pdfout("<<\n");
			pdfout("  /Type /Font\n");
			if (fonttype(e->path) == 't')
				pdfout("  /Subtype /TrueType\n");
			else
				pdfout("  /Subtype /Type1\n");
			pdfout("  /BaseFont /%s\n", e->name);
			pdfout("  /FirstChar %d\n", e->minc % 256);
			pdfout("  /LastChar %d\n", e->maxc % 256);
			pdfout("  /Widths [");
			for (k = e->minc; k <= e->maxc; k++) {
				g = font_glget(f, k);
				pdfout(" %d", (long)g->width * 0x1c20 / dev_res);
			}
			pdfout(" ]\n");
			pdfout("  /FontDescriptor %d 0 R\n", e->wid);
			pdfout("  /Encoding %d 0 R\n", cid);
			pdfout(">>\n");
			pdfout("endobj\n\n");
			font_close(f);
		}
	}
	free(pfonts);

	id = obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	if (pdf_title[0])
		pdfout("  /Title %s\n", pdftext_static(pdf_title));
	if (pdf_author[0])
		pdfout("  /Author %s\n", pdftext_static(pdf_author));
	pdfout("  /Creator (Neatroff)\n");
	pdfout("  /Producer (Neatpost)\n");
	pdfout(">>\n");
	pdfout("endobj\n\n");

	xref = pdf_pos;
	pdfout("xref\n");
	pdfout("0 %d\n", obj_n);
	pdfout("0000000000 65535 f \n");
	for (i = 1; i < obj_n; i++)
		pdfout("%010d 00000 n \n", obj_off[i]);
	pdfout("trailer\n");
	pdfout("<<\n");
	pdfout("  /Size %d\n", obj_n);
	pdfout("  /Root %d 0 R\n", pdf_root);
	pdfout("  /Info %d 0 R\n", id);
	pdfout(">>\n");
	pdfout("startxref\n");
	pdfout("%d\n", xref);
	pdfout("%%%%EOF\n");
	free(page_id);
	free(obj_off);
}
