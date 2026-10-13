#include <string.h>
#include "wb.h"
#include "c_ec.h"
#include "wb_flushsub.h"
#include "wb_putbuf.h"
#include "fn_dev_font.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_hy.h"
#include "fn_font_wid.h"
#include "fn_font_find.h"
#include "font_layout.h"
#include "env_hc.h"
#include "fn_sbuf_printf.h"

static void wb_flushsub(struct wb *w)
{
	char hyf[256];
	int pos[256];
	int xa[256], xb[256], xc[256], xd[256];
	void *f;
	int hyv, n, i, j, k, cnt, cnt2, kn, lg, wid, len, dir;
	char *out, *e;

	if (w->f2068 && w->f206c) {
		w->f206c = 0;
		f = dev_font(w->f10[0]);
		if (*nreg(map(".hy"))) {
			char text[0x2000];
			char flags[0x2000];

			hyv = *nreg(map(".hy"));
			n = w->f2068;
			text[0] = 0;
			out = text;
			if (n > 0) {
				for (i = 0; i < n; i++) {
					e = (char *)w + 0x68 + i * 32;
					pos[i] = out - text;
					if (c_hystop(e))
						break;
					if (!c_hymark(e))
						out += hy_cput(out, e);
				}
				if (i == n) {
					len = out - text;
					memset(flags, 0, len);
					hyphenate(flags, text, hyv);
					for (i = 0; i < n; i++)
						hyf[i] = flags[pos[i]];
				} else
					memset(hyf, 0, 256);
			} else
				hyphenate(flags, text, hyv);
		} else
			memset(hyf, 0, 256);
		for (i = 0; i < w->f2068; ) {
			void *fp[1024];
			void *gl[1024];
			int h[2], v[2];

			j = i;
			cnt = 0;
			for (; j < w->f2068; j++) {
				e = (char *)w + 0x68 + j * 32;
				if (c_hymark(e))
					break;
				fp[cnt++] = font_find(f, e);
			}
			kn = *nreg(map(".kn"));
			lg = *nreg(map(".lg"));
			cnt2 = font_layout(f, fp, cnt, w->f10[1], gl, pos, xa, xb, xc, xd, lg, kn);
			if (cnt2 > 0) {
				for (k = 0; k < cnt2; k++) {
					h[0] = xa[k];
					h[1] = xc[k] - xa[k];
					v[0] = xb[k];
					v[1] = xd[k] - xb[k];
					dir = w->f10[3];
					if (h[dir]) {
						wid = font_wid(f, w->f10[1]);
						if (w->f2068 && w->f206c)
							wb_flushsub(w);
						w->hpos += wid;
						sbuf_printf(&w->buf, "%ch'%du'", c_ec, wid);
					}
					if (v[dir]) {
						wid = font_wid(f, w->f10[1]);
						if (w->f2068 && w->f206c)
							wb_flushsub(w);
						w->vpos += wid;
						sbuf_printf(&w->buf, "%cv'%du'", c_ec, wid);
					}
					if (hyf[pos[k] + i])
						wb_putbuf(w, env_hc());
					if (gl[k] == fp[pos[k]])
						wb_putbuf(w, (char *)w + 0x68 + (pos[k] + i) * 32);
					else
						wb_putbuf(w, (char *)gl[k] + 0x20);
					dir = 1 - w->f10[3];
					if (h[dir]) {
						wid = font_wid(f, w->f10[1]);
						if (w->f2068 && w->f206c)
							wb_flushsub(w);
						w->hpos += wid;
						sbuf_printf(&w->buf, "%ch'%du'", c_ec, wid);
					}
					if (v[dir]) {
						wid = font_wid(f, w->f10[1]);
						if (w->f2068 && w->f206c)
							wb_flushsub(w);
						w->vpos += wid;
						sbuf_printf(&w->buf, "%cv'%du'", c_ec, wid);
					}
				}
			}
			i = j;
			while (i < w->f2068) {
				e = (char *)w + 0x68 + i * 32;
				if (!c_hymark(e))
					break;
				wb_putbuf(w, e);
				i++;
			}
		}
		w->f2068 = 0;
		w->f206c = 1;
		*(int *)w->pad64 = 0;
	}
}
