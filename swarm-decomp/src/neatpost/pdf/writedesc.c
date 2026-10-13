#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "fn_font_path.h"
#include "fn_sbuf_buf.h"
#include "fn_sbuf_chr.h"
#include "fn_sbuf_cut.h"
#include "fn_sbuf_free.h"
#include "fn_sbuf_len.h"
#include "fn_sbuf_make.h"
#include "fn_sbuf_mem.h"
#include "fn_sbuf_str.h"
#include "font_name.h"
#include "fonttype.h"
#include "obj_map.h"
#include "obj_off.h"
#include "pdf_pos.h"
#include "pdfout.h"
#include "writedesc.h"

static int writedesc(struct font *f)
{
	static const char hex[] = "0123456789ABCDEF";
	struct sbuf *raw, *hx;
	char *p, *q;
	char buf[1024];
	int type, ffid, id, n, len, l1, l2, i;

	type = fonttype(font_path(f));
	ffid = -1;
	if (type == '1' || type == 't') {
		ffid = open(font_path(f), 0);
		raw = sbuf_make();
		hx = sbuf_make();
		while ((n = read(ffid, buf, sizeof buf)) > 0)
			sbuf_mem(raw, buf, n);
		close(ffid);
		l1 = sbuf_len(raw);
		l2 = 0;
		if (type == '1') {
			len = sbuf_len(raw);
			p = sbuf_buf(raw);
			if (len > 5) {
				for (i = 0; i < len - 5; i++)
					if (p[i] == 'e' && memcmp(p + i, "eexec", 5) == 0)
						break;
				if (i == len - 5) {
					l1 = 0;
					l2 = 0;
				} else {
					l1 = i;
					for (n = i + 1; n < len - 512; n++)
						if (p[n] == '0' && memcmp(p + n, "00000", 5) == 0)
							break;
					if (n < len - 512) {
						if (len - n != 0)
							sbuf_cut(raw, i + n);
						l2 = n;
						l1 = i - (len - n);
					}
				}
			} else {
				l1 = 0;
			}
		}
		len = sbuf_len(raw);
		p = sbuf_buf(raw);
		for (i = 0; i < len; i++) {
			sbuf_chr(hx, hex[(unsigned char)p[i] >> 4 & 15]);
			sbuf_chr(hx, hex[(unsigned char)p[i] & 15]);
			if (i % 40 == 39 && i + 1 < len)
				sbuf_chr(hx, '\n');
		}
		ffid = -1;
		sbuf_str(hx, ">\n");
		if (l1 != 0) {
			ffid = obj_map();
			obj_off[ffid] = pdf_pos;
			pdfout("%d 0 obj\n", ffid);
			pdfout("<<\n");
			pdfout("  /Filter /ASCIIHexDecode\n");
			pdfout("  /Length %d\n", sbuf_len(hx));
			pdfout("  /Length1 %d\n", l1);
			if (type == '1') {
				pdfout("  /Length2 %d\n", l2);
				pdfout("  /Length3 %d\n", 0);
			}
			pdfout(">>\n");
			pdfout("stream\n");
			n = sbuf_len(hx);
			q = sbuf_buf(hx);
			fwrite(q, n, 1, stdout);
			pdf_pos += n;
			pdfout("endstream\n");
			pdfout("endobj\n\n");
		}
		sbuf_free(raw);
		sbuf_free(hx);
	}
	id = obj_map();
	obj_off[id] = pdf_pos;
	pdfout("%d 0 obj\n", id);
	pdfout("<<\n");
	pdfout("  /Type /FontDescriptor\n");
	pdfout("  /FontName /%s\n", font_name(f));
	pdfout("  /Flags 32\n");
	pdfout("  /FontBBox [-1000 -1000 1000 1000]\n");
	pdfout("  /MissingWidth 1000\n");
	pdfout("  /StemV 100\n");
	pdfout("  /ItalicAngle 0\n");
	pdfout("  /CapHeight 100\n");
	pdfout("  /Ascent 100\n");
	pdfout("  /Descent 100\n");
	if (ffid >= 0)
		pdfout("  /FontFile%s %d 0 R\n", type == 't' ? "2" : "", ffid);
	pdfout(">>\n");
	pdfout("endobj\n\n");
	return id;
}
