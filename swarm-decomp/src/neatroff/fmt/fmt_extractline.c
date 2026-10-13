#include <string.h>
#include "fmt.h"
#include "fn_mextend.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_printf.h"
#include "fn_free.h"
#include "fn_nreg.h"
#include "fn_map.h"
#include "c_ec.h"
#include "fmt_keshideh.h"
#include "fmt_extractline.h"

/* not yet PASS (insn-delta mid, frac 0.82): the justify divisor is the
   f30 span of the words, not n; gcc still duplicates the ext/sp block and
   the l->f1c store / hy reload order and the reg split (avail vs n*56) differ */
static int fmt_extractline(struct fmt *f, int n, int b)
{
	struct fmtline *l;
	struct fmtword *w;
	int k, i, avail, linewid, hy, wid, ext, q, rem, sp;

	if (f->nlines == f->curline) {
		f->nlines = 0;
		f->curline = 0;
	}
	k = f->nlines;
	if (f->linecap == k) {
		f->linecap = k + 0x100;
		f->lines = mextend(f->lines, k, f->linecap, sizeof(struct fmtline));
		k = f->nlines;
	}
	f->nlines = k + 1;
	l = &f->lines[k];
	l->f14 = f->f4c;
	l->f18 = f->f50;
	l->f1c = f->f54;
	sbuf_init(&l->buf);
	avail = f->f50 - f->f4c - f->f54;
	if (avail < 0)
		avail = 0;
	wid = 0;
	if (n != 0)
		wid = f->words[n - 1].pos - f->words[0].pos + f->words[0].wid + f->words[n - 1].f18;
	if (b && *nreg('u')) {
		if (!*nreg(map(".na")) && !*nreg(map(".ce"))) {
			if ((*nreg('j') & 0x10) && (*nreg('j') & 0x40)) {
				fmt_keshideh(f, n, avail - wid);
				if (n != 0)
					wid = f->words[n - 1].pos - f->words[0].pos + f->words[0].wid + f->words[n - 1].f18;
			}
		}
	}
	if (n != 0) {
		sp = f->words[n - 1].f30 - f->words[0].f30;
		if (sp && *nreg('u')) {
			if (!*nreg(map(".na")) && !*nreg(map(".ce")) && (*nreg('j') & 0x10) && ((b & 1) || avail < wid)) {
				ext = avail - wid;
				q = ext / sp;
				rem = ext % sp;
				if (rem < 0) {
					q--;
					rem += sp;
				}
				for (i = 1; i < n; i++) {
					if (f->words[i].f1c != 0) {
						f->words[i].f14 += q + (rem > 0);
						rem--;
					}
				}
			}
		}
	}
	l->f20 = 0;
	l->f24 = 0;
	linewid = 0;
	for (i = 0; i < n; i++) {
		w = &f->words[i];
		if (w->f14)
			sbuf_printf(&l->buf, "%ch'%du'", c_ec, w->f14);
		sbuf_append(&l->buf, w->s);
		linewid += w->wid + w->f14;
		if (w->f0c < l->f20)
			l->f20 = w->f0c;
		if (w->f10 > l->f24)
			l->f24 = w->f10;
		free(w->s);
	}
	hy = n > 0 ? f->words[n - 1].f18 : 0;
	if (hy)
		sbuf_append(&l->buf, "\\(hy");
	f->fillreq -= n;
	f->nwords -= n;
	memmove(f->words, f->words + n, f->nwords * sizeof(struct fmtword));
	l->f10 = hy + linewid;
	return 0;
}
