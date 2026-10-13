#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include "obj_map.h"
#include "obj_off.h"
#include "o_queued.h"
#include "out_fontup.h"
#include "pdf_pos.h"
#include "pdfout.h"
#include "pdfpos00.h"
#include "pdf_copy.h"
#include "pdf_dictcopy.h"
#include "pdf_strcopy.h"
#include "pg.h"
#include "xobj.h"
#include "xobj_n.h"
#include "fn_pdf_dval_obj.h"
#include "fn_pdf_dval_val.h"
#include "fn_pdf_lval.h"
#include "fn_pdf_ref.h"
#include "fn_pdf_trailer.h"
#include "fn_pdf_type.h"
#include "fn_sbuf_make.h"
#include "fn_sbuf_free.h"
#include "fn_sbuf_printf.h"

void outpdf(char *s, int w, int h)
{
	struct sbuf *sb, *sb2;
	char buf[4096];
	char *data;
	char *fk[2];
	char *rk[4];
	int bb[4];
	int fd, n, end, tr, root, pages, kids, lv, ref, contents, lenv, res, len, mb;
	int id, idx, i, j, k, c, v, fr, sc, x, y, wi, wf, hi, hf;

	sb = sbuf_make();
	fd = open(s, 0);
	while ((n = read(fd, buf, sizeof buf)) > 0)
		sbuf_mem(sb, buf, n);
	close(fd);
	end = sbuf_len(sb);
	data = sbuf_buf(sb);
	fk[0] = "/Filter";
	fk[1] = "/DecodeParms";
	if (xobj_n == 0x80)
		goto done;
	tr = pdf_trailer(data, end);
	if (tr < 0)
		goto done;
	root = pdf_dval_obj(data, end, tr, "/Root");
	if (root < 0)
		goto done;
	pages = pdf_dval_obj(data, end, root, "/Pages");
	if (pages < 0)
		goto done;
	kids = pdf_dval_val(data, end, pages, "/Kids");
	if (kids < 0)
		goto done;
	lv = pdf_lval(data, end, kids, 0);
	if (lv < 0)
		goto done;
	ref = pdf_ref(data, end, lv);
	if (ref < 0)
		goto done;
	contents = pdf_dval_obj(data, end, ref, "/Contents");
	if (contents < 0)
		goto done;
	lenv = pdf_dval_val(data, end, contents, "/Length");
	if (lenv < 0)
		goto done;
	res = pdf_dval_val(data, end, ref, "/Resources");
	len = strtol(data + lenv, 0, 10);
	mb = pdf_dval_val(data, end, ref, "/MediaBox");
	if (mb < 0)
		mb = pdf_dval_val(data, end, pages, "/MediaBox");
	if (mb >= 0) {
		for (i = 0; i < 4; i++) {
			k = pdf_lval(data, end, mb, i);
			if (k < 0)
				goto defaults;
			c = data[k];
			v = 0;
			if (isdigit(c)) {
				j = k;
				do {
					v = v * 10 + (c - '0');
					c = data[++j];
				} while (isdigit(c));
			}
			fr = 0;
			if (c == '.' && isdigit(data[j + 1])) {
				fr = (data[j + 1] - '0') * 10;
				if (isdigit(data[j + 2]))
					fr += data[j + 2] - '0';
			}
			bb[i] = v * 100 + fr;
		}
		sc = 720000 / dev_res;
		if (w > 0) {
			x = (long)w * sc / (bb[2] - bb[0]);
			wi = x / 100;
			wf = x % 100;
			if (h > 0) {
				y = (long)h * sc / (bb[3] - bb[1]);
				hi = y / 100;
				hf = y % 100;
			} else {
				hi = wi;
				hf = wf;
			}
		} else if (h > 0) {
			y = (long)h * sc / (bb[3] - bb[1]);
			wi = hi = y / 100;
			wf = hf = y % 100;
		} else {
			goto defaults;
		}
		goto bbox_out;
defaults:
		wi = 1;
		wf = 0;
		hi = 1;
		hf = 0;
bbox_out:
		sb2 = sbuf_make();
		sbuf_printf(sb2, "<<\n");
		sbuf_printf(sb2, "  /Type /XObject\n");
		sbuf_printf(sb2, "  /Subtype /Form\n");
		sbuf_printf(sb2, "  /FormType 1\n");
		sbuf_printf(sb2, "  /BBox %s\n", pdf_copy(data, end, mb));
	} else {
		sb2 = sbuf_make();
		sbuf_printf(sb2, "<<\n");
		sbuf_printf(sb2, "  /Type /XObject\n");
		sbuf_printf(sb2, "  /Subtype /Form\n");
		sbuf_printf(sb2, "  /FormType 1\n");
		wi = 1;
		wf = 0;
		hi = 1;
		hf = 0;
	}
	sbuf_printf(sb2, "  /Matrix [%d.%02d 0 0 %d.%02d %s]\n", wi, wf, hi, hf,
		pdfpos00(o_h * 100, o_v * 100));
	if (res >= 0) {
		sbuf_printf(sb2, "  /Resources <<\n");
		for (i = 0; i < 4; i++) {
			v = pdf_dval_val(data, end, res, rk[i]);
			if (v < 0)
				continue;
			if (pdf_type(data, end, v) == 0x64) {
				sbuf_printf(sb2, "    %s ", rk[i]);
				pdf_dictcopy(data, end, v, sb2);
				sbuf_printf(sb2, "\n");
			} else {
				sbuf_printf(sb2, "    %s %s\n", rk[i], pdf_copy(data, end, v));
			}
		}
		sbuf_printf(sb2, "  >>\n");
	}
	sbuf_printf(sb2, "  /Length %d\n", len);
	for (i = 0; i < 2; i++) {
		v = pdf_dval_val(data, end, contents, fk[i]);
		if (v >= 0)
			sbuf_printf(sb2, "  %s %s\n", fk[i], pdf_copy(data, end, v));
	}
	sbuf_printf(sb2, ">>\n");
	pdf_strcopy(data, end, contents, sb2);
	id = obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	fwrite(sbuf_buf(sb2), sbuf_len(sb2), 1, stdout);
	pdf_pos += sbuf_len(sb2);
	pdfout("endobj\n\n");
	sbuf_free(sb2);
	idx = xobj_n;
	xobj_n = idx + 1;
	xobj[idx] = id;
	sbuf_free(sb);
	if (o_queued == 1)
		sbuf_printf(pg, ">] TJ\n");
	o_queued = 0;
	out_fontup();
	if (idx >= 0)
		sbuf_printf(pg, "ET /FO%d Do BT\n", idx);
	p_h = -1;
	p_v = -1;
	return;
done:
	sbuf_free(sb);
	if (o_queued == 1)
		sbuf_printf(pg, ">] TJ\n");
	o_queued = 0;
	out_fontup();
	p_h = -1;
	p_v = -1;
}
