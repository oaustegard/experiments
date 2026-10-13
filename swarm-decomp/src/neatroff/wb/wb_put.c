#include "wb.h"
#include "c_ec.h"
#include "wb_flushdir.h"
#include "wb_flushfont.h"
#include "wb_flushsub.h"
#include "wb_putbuf.h"
#include "fn_dev_font.h"
#include "fn_font_find.h"
#include "fn_hy.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_printf.h"
#include "fn_wb_put.h"
#include "fn_font_swid.h"

void wb_put(struct wb *w, char *s)
{
	if (*s != '\n') {
	int fam, fn, sz, ms, saved;
	void *f;

	if (*s == '\n') {
		w->part = 0;
		return;
	}
	fam = w->f20[3] < 0 ? *nreg(map(".cd")) : w->f20[3];
	if (w->f10[3] != fam)
		wb_flushdir(w);
	if (*s == ' ') {
		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		{
			int ss = *nreg(map(".ss"));
			int sz = w->f20[1];
			int fn2 = w->f20[0];

			if (sz < 0)
				sz = *nreg('s');
			if (fn2 < 0)
				fn2 = *nreg('f');
			fn = font_swid(dev_font(fn2), sz, ss);
		}
		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		w->hpos += fn;
		sbuf_printf(&w->buf, "%ch'%du'", c_ec, fn);
		return;
	}
	if (s[0] == '\\' && s[1] == '~' && s[2] == 0) {
		if (w->f2068 && w->f206c)
			wb_flushsub(w);
		sbuf_append(&w->buf, s);
		{
			int ss = *nreg(map(".ss"));
			int sz = w->f20[1];
			int fn2 = w->f20[0];

			if (sz < 0)
				sz = *nreg('s');
			if (fn2 < 0)
				fn2 = *nreg('f');
			w->hpos += font_swid(dev_font(fn2), sz, ss);
		}
		return;
	}
	fn = w->f20[0] < 0 ? *nreg('f') : w->f20[0];
	if (w->f10[0] == fn) {
		sz = w->f20[1] < 0 ? *nreg('s') : w->f20[1];
		if (sz == w->f10[1]) {
			if (!*nreg('C')) {
				ms = w->f20[2] < 0 ? *nreg('m') : w->f20[2];
				if (ms != w->f10[2])
					goto flush;
			}
			if (w->f2068 == 0x100)
				goto flush;
			goto append;
		}
	}
flush:
	if (w->f2068 && w->f206c)
		wb_flushsub(w);
	wb_flushdir(w);
	wb_flushfont(w);
append:
	if (w->f206c == 0) {
		wb_putbuf(w, s);
		return;
	}
	f = dev_font(w->f10[0]);
	if (font_find(f, s)) {
		w->f2068++;
		strcpy((char *)w + 0x68 + (w->f2068 - 1) * 32, s);
		return;
	}
	if (c_hymark(s)) {
		w->f2068++;
		strcpy((char *)w + 0x68 + (w->f2068 - 1) * 32, s);
		return;
	}
	saved = w->f206c;
	if (w->f2068 && w->f206c) {
		wb_flushsub(w);
		saved = w->f206c;
	}
	w->f206c = 0;
	if (*s == '\n')
		w->part = 0;
	else
		wb_put(w, s);
	w->f206c = saved;
	} else
		w->part = 0;
}
