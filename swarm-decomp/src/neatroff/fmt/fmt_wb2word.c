#include <string.h>
#include "fmt.h"
#include "wb.h"
#include "fn_wb_buf.h"
#include "fn_wb_wid.h"
#include "fn_wb_swid.h"
#include "fn_wb_hywid.h"
#include "fn_xmalloc.h"
#include "fn_mextend.h"

static void fmt_wb2word(struct fmt *f, struct wb *w, int hy, int a4, int a5, int a6)
{
	struct fmtword *p, *prev;
	int n;

	n = strlen(wb_buf(w));
	if (f->nwords == f->ncap) {
		f->ncap = f->nwords + 256;
		f->words = mextend(f->words, f->nwords, f->ncap, sizeof(struct fmtword));
	}
	p = &f->words[f->nwords++];
	prev = f->nwords > 1 ? &f->words[f->nwords - 2] : 0;
	p->s = xmalloc(n + 1);
	memcpy(p->s, wb_buf(w), n + 1);
	p->wid = wb_wid(w);
	p->f0c = w->a38[0];
	p->f10 = w->a38[1];
	if (hy)
		hy = wb_hywid(w);
	p->f14 = a5;
	p->f18 = hy;
	p->f1c = a4;
	p->f20 = a6;
	p->swid = wb_swid(w);
	if (prev) {
		p->pos = prev->pos + p->wid + p->f14;
		p->f2c = a4 ? a5 + prev->f2c : prev->f2c;
		p->f30 = a4 + prev->f30;
	} else {
		p->pos = p->wid + p->f14;
		p->f2c = a4 ? a5 : 0;
		p->f30 = a4;
	}
}
