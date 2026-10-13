#include <string.h>
#include "wb.h"
#include "glyph.h"
#include "wb_putc.h"
#include "fn_wb_put.h"
#include "fn_wb_reset.h"
#include "fn_dev_glyph.h"
#include "fn_dev_font.h"
#include "fn_font_gwid.h"
#include "fn_escread.h"
#include "fn_nreg.h"

int wb_keshideh(char *s, struct wb *w, int n)
{
	char name[32] = {0};
	struct glyph *g;
	int ret = 0, c, wid, sz, fn, kw;
	char *x, *start = s;
	char *prev = 0, *best = 0;
	const char *const *p;
	const char *src;
	while ((c = escread(&s, &x)) >= 0) {
		wb_putc(w, c, x);
		if (c != 0) {
			src = "";
		} else {
			for (p = keshideh_chars; p < keshideh_chars + 28; p++) {
				if (strcmp(*p, name) == 0) {
					g = dev_glyph("\xd9\x80", w->f20[0] < 0 ? *nreg('f') : w->f20[0]);
					if (g != 0) {
						wid = g->a48;
						sz = w->f20[1] < 0 ? *nreg('s') : w->f20[1];
						fn = w->f20[0] < 0 ? *nreg('f') : w->f20[0];
						kw = font_gwid(g->font, dev_font(fn), sz, wid);
						if (kw < n) {
							ret = kw;
							best = prev;
						}
					}
					break;
				}
			}
			src = x;
		}
		prev = s;
		strcpy(name, src);
	}
	s = start;
	wb_reset(w);
	while ((c = escread(&s, &x)) >= 0) {
		wb_putc(w, c, x);
		if (s == best)
			wb_put(w, "\xd9\x80");
	}
	return ret;
}
