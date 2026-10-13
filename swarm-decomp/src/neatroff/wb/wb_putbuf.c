#include <string.h>
#include "wb.h"
#include "glyph.h"
#include "c_ec.h"
#include "var_dev_res.h"
#include "wb_flushsub.h"
#include "wb_putbuf.h"
#include "fn_dev_font.h"
#include "fn_dev_glyph.h"
#include "fn_font_gwid.h"
#include "fn_hy.h"
#include "fn_nreg.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_printf.h"
#include "fn_utf8one.h"

/* font_wid is declared unprototyped in types/; calls here use its real 3-arg type */
#define font_wid3(f, n, k) ((int (*)(void *, int, int))font_wid)(f, n, k)

static void wb_putbuf(struct wb *w, char *s)
{
	struct glyph *g;
	int hy, C, wid, a, hd, A, B, D, Cw, t;

	if ((s[0] & ~8) == 1 || (s[0] == 4 && (s[1] & ~8) == 1)) {
		sbuf_append(&w->buf, s);
		return;
	}
	g = dev_glyph(s, w->f10[0]);
	hy = c_hymark(s);
	if (!g && *s == c_ec && hy == 0) {
		memmove(s, s + 1, strlen(s));
		g = dev_glyph(s, w->f10[0]);
	}
	if (g && hy == 0 && *(int *)w->pad64 && g->a4a[0] < 0) {
		int n = font_wid3(g->font, w->f10[1], -g->a4a[0]);

		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		w->hpos += n;
		sbuf_printf(&w->buf, "%ch'%du'", c_ec, n);
	}
	*(int *)w->pad64 = 0;
	if (s[1] != 0 && s[0] != c_ec && s[0] != 4 && !utf8one(s)) {
		if (s[1] != 0 && s[2] == 0)
			sbuf_printf(&w->buf, "%c(%s", c_ec, s);
		else
			sbuf_printf(&w->buf, "%cC'%s'", c_ec, s);
	} else if (s[0] == 4 && s[1] == c_ec) {
		int c = s[1];

		sbuf_printf(&w->buf, "%c%c", c, c);
	} else
		sbuf_append(&w->buf, s);
	if (!hy) {
		C = *nreg('C');
		if (!C) {
			if (g) {
				if (g->a4a[0] || g->a4a[1] || g->a4e || g->a50) {
					A = font_wid3(g->font, w->f10[1], g->a4a[0]);
					B = font_wid3(g->font, w->f10[1], g->a4a[1]);
					Cw = font_wid3(g->font, w->f10[1], g->a4e);
					D = font_wid3(g->font, w->f10[1], g->a50);
					w->a54[0] = w->hpos + A <= w->a54[0] ? w->hpos + A : w->a54[0];
					w->a54[1] = B - w->vpos > w->a54[1] ? w->a54[1] : B - w->vpos;
					w->a54[2] = w->hpos + Cw < w->a54[2] ? w->a54[2] : w->hpos + Cw;
					w->a54[3] = D - w->vpos < w->a54[3] ? w->a54[3] : D - w->vpos;
				} else {
					a = dev_res / 72 * w->f10[1];
					wid = font_wid3(g->font, w->f10[1], g->a48);
					hd = 0;
					t = (unsigned short)g->a52;
					if (t & 1)
						hd = -(a / 2);
					if (!(t & 2))
						a = a / 2;
					w->a54[0] = w->hpos <= w->a54[0] ? w->hpos : w->a54[0];
					w->a54[1] = hd - w->vpos > w->a54[1] ? w->a54[1] : hd - w->vpos;
					w->a54[2] = w->hpos + wid < w->a54[2] ? w->a54[2] : w->hpos + wid;
					w->a54[3] = a - w->vpos < w->a54[3] ? w->a54[3] : a - w->vpos;
				}
			}
		}
		if (g) {
			w->hpos += font_gwid(g->font, dev_font(w->f10[0]), w->f10[1], g->a48);
			w->a48[0] |= g->a52;
		}
		a = w->f10[1] * dev_res / 72;
		t = w->vpos + a;
		w->a48[2] = t > w->a48[2] ? w->a48[2] : t;
		w->a48[1] = w->a48[1] < w->vpos ? w->vpos : w->a48[1];
	}
}
