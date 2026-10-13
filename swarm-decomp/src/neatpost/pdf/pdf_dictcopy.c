#include <stdio.h>
#include "fn_pdf_dkey.h"
#include "fn_pdf_dval.h"
#include "fn_pdf_type.h"
#include "fn_pdf_ref.h"
#include "fn_pdf_len.h"
#include "fn_sbuf_printf.h"
#include "fn_sbuf_make.h"
#include "fn_sbuf_chr.h"
#include "fn_sbuf_len.h"
#include "fn_sbuf_buf.h"
#include "fn_sbuf_free.h"
#include "pdf_copy.h"
#include "pdf_strcopy.h"
#include "obj_map.h"
#include "obj_off.h"
#include "pdf_pos.h"
#include "pdfout.h"
#include "pdf_dictcopy.h"

static void pdf_dictcopy(unsigned char *s, int end, int i, struct sbuf *sb)
{
	struct sbuf *sb2;
	int n, k, v, ref, id, len;

	n = 0;
	sbuf_printf(sb, "<<");
	for (;;) {
		k = pdf_dkey(s, end, i, n);
		if (k < 0)
			break;
		sbuf_printf(sb, " %s", pdf_copy(s, end, k));
		v = pdf_dval(s, end, i, pdf_copy(s, end, k));
		if (pdf_type(s, end, v) != 0x72) {
			sbuf_printf(sb, " %s", pdf_copy(s, end, v));
		} else {
			ref = pdf_ref(s, end, v);
			if (ref >= 0) {
				if (pdf_type(s, end, ref) == 0x64) {
					sb2 = sbuf_make();
					pdf_dictcopy(s, end, ref, sb2);
					sbuf_chr(sb2, '\n');
					if (pdf_dval(s, end, ref, "/Length") >= 0)
						pdf_strcopy(s, end, ref, sb2);
					id = obj_map();
					obj_off[id] = pdf_pos;
					pdfout("%d 0 obj\n", id);
					len = sbuf_len(sb2);
					fwrite(sbuf_buf(sb2), len, 1, stdout);
					pdf_pos += len;
					pdfout("endobj\n\n");
					sbuf_free(sb2);
				} else {
					id = obj_map();
					obj_off[id] = pdf_pos;
					pdfout("%d 0 obj\n", id);
					len = pdf_len(s, end, ref);
					fwrite(s + ref, len, 1, stdout);
					pdf_pos += len;
					pdfout("\n");
					pdfout("endobj\n\n");
				}
				if (id >= 0)
					sbuf_printf(sb, " %d 0 R", id);
			}
		}
		n++;
	}
	sbuf_printf(sb, " >>");
}
