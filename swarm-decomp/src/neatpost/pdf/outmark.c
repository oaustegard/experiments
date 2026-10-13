#include <stdlib.h>
#include "fn_pdftext_static.h"
#include "fn_mextend.h"
#include "fn_sbuf_printf.h"
#include "dev_res.h"
#include "obj_map.h"
#include "pdf_outline.h"
#include "o_queued.h"
#include "pg.h"
#include "obj_off.h"
#include "pdf_pos.h"
#include "page_n.h"
#include "page_id.h"
#include "pdf_height.h"

/* Outline (bookmark) tree: n marks, each with a title, a destination page and
 * y position, and an outline level. */
void outmark(int n, char (*names)[256], int *pages, int *ys, int *levels)
{
	int *ids;
	int j, id, cnt;
	long i, last, prev;

	ids = malloc(n * sizeof(int));
	pdf_outline = obj_map();
	for (i = 0; i < n; i++)
		ids[i] = obj_map();
	if (o_queued == 1)
		sbuf_printf(pg, ">] TJ\n");
	o_queued = 0;
	id = pdf_outline > 0 ? pdf_outline : obj_map();
	obj_off[id] = pdf_pos;
	last = n - 1;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	if (n > 0) {
		cnt = 0;
		for (i = 0; i < n; i++)
			if (levels[i] == levels[0])
				cnt++;
		pdfout("  /Count %d\n", cnt);
		pdfout("  /First %d 0 R\n", ids[0]);
		while (last > 0 && levels[last] > levels[0])
			last--;
		pdfout("  /Last %d 0 R\n", ids[last]);
	} else {
		pdfout("  /Count %d\n", 0);
		pdfout("  /First %d 0 R\n", ids[0]);
		pdfout("  /Last %d 0 R\n", ids[n - 1]);
	}
	pdfout(">>\n");
	pdfout("endobj\n\n");

	prev = -1;
	for (i = 0; i < n; i++) {
		cnt = 0;
		for (j = i + 1; j < n && levels[j] > levels[i]; j++)
			if (levels[j] == levels[i] + 1)
				cnt++;
		id = ids[i] > 0 ? ids[i] : obj_map();
		obj_off[id] = pdf_pos;
		pdfout("%d 0 obj\n", id);
		pdfout("<<\n");
		pdfout("  /Title %s\n", pdftext_static(names[i]));
		if (prev >= 0) {
			for (j = prev; j >= 0 && levels[j] >= levels[i]; j--)
				;
			pdfout("  /Parent %d 0 R\n", j < 0 ? pdf_outline : ids[j]);
		} else
			pdfout("  /Parent %d 0 R\n", pdf_outline);
		for (j = i + 1; j < n && levels[j] > levels[i]; j++)
			;
		if (j < n && levels[j] == levels[i])
			pdfout("  /Next %d 0 R\n", ids[j]);
		if (prev >= 0) {
			for (j = prev; j >= 0 && levels[j] > levels[i]; j--)
				;
			if (j >= 0 && levels[j] == levels[i])
				pdfout("  /Prev %d 0 R\n", ids[j]);
		}
		if (cnt != 0) {
			pdfout("  /Count %d\n", cnt);
			pdfout("  /First %d 0 R\n", ids[i + 1]);
			last = 0;
			for (j = i + 1; j < n && levels[j] > levels[i]; j++)
				if (levels[j] == levels[i] + 1)
					last = j;
			pdfout("  /Last %d 0 R\n", ids[last]);
		}
		if (pages[i] > 0 && pages[i] <= page_n)
			pdfout("  /Dest [ %d 0 R /XYZ 0 %d 0 ]\n", page_id[pages[i] - 1],
			       pdf_height - 72 * ys[i] / dev_res);
		pdfout(">>\n");
		pdfout("endobj\n\n");
		prev = i;
	}
	free(ids);
}
