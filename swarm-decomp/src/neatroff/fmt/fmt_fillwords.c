#include "fmt.h"
#include "fmt_fillwords.h"
#include "fmtb_cost.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_f_nexttrap.h"
#include "fn_xmalloc.h"
#include "fn_free.h"
#include "fn_memset.h"

/* partial: the setup, the fmt_safelines() decisions, the hyphenation-cost
   table and the allocation of the DP arrays are written; the line-breaking
   DP itself (which calls fmta_break and fmtb_cost, both still stubs in this
   TU) is not yet matched, see notes.  The calls to fmt_confchanged,
   fmt_confupdate and fmt_extractline are kept so that gcc does not inline or
   clone those helpers differently in their other callers. */
static int fmt_fillwords(struct fmt *f, int force)
{
	struct fmtword *w, *last;
	int ce, avail, lim, n, nofill, flag, x, ssh, hlm, hj, nl, m;
	int width, i;
	int a74, a78, hyc[3];
	long *costs;
	long cost, c;
	int *B, *C;
	int brk[0x40], nbrk;

	ce = *nreg(map(".ce"));
	if (ce)
		return 0;
	if (*nreg('u') == 0)
		return 0;
	if (f->nwords == 0)
		return 0;

	w = f->words;
	last = &f->words[f->nwords - 1];
	ssh = *nreg(map(".ssh"));
	width = last->pos - w->pos + w->wid + last->f18;
	avail = f->f50 - f->f4c - f->f54;
	if (avail < 0)
		avail = 0;

	lim = f->fillreq;
	if (lim > 0 && lim <= f->nwords) {
		nofill = 0;
		flag = !force;
	} else {
		if (!force) {
			x = width + ((last->f2c - w->f2c) * ssh) / 100;
			if (avail >= x)
				return 0;
		}
		lim = -1;
		nofill = 1;
		flag = !force;
	}

	n = fmt_safelines() - f->nlines + f->curline;
	if (n <= 0 && fmt_safelines() > 0)
		return 1;
	if (fmt_safelines() != 0) {
		a78 = fmt_confchanged(f) ? 1 : n;
		a74 = (unsigned)n <= 0x40 ? n : 0x40;
		if (n <= 0) {
			n = -1;
			goto hy_done;
		}
	} else {
		fmt_confchanged(f);
		a74 = 1;
		a78 = 1;
		n = 1;
	}
	if (!(*nreg(map(".hy")) & 2))
		n = -1;
hy_done:
	hj = *nreg('j') & 0x80;
	if (!hj) {
		avail = f->f50 - f->f4c - f->f54;
		if (avail < 0)
			avail = 0;
		nl = lim;
		if (nl <= 0)
			nl = f->nwords;
		ssh = *nreg(map(".ssh"));
		hlm = *nreg(map(".hlm"));
		hyc[0] = *nreg(map(".hycost"));
		hyc[1] = *nreg(map(".hycost")) + *nreg(map(".hycost2"));
		hyc[2] = *nreg(map(".hycost")) + *nreg(map(".hycost2"))
		    + *nreg(map(".hycost3"));
		(void)hlm;
		(void)hyc;
		m = nl + 1;
		costs = xmalloc((long)m * 8);
		B = xmalloc((long)m * 4);
		C = xmalloc((long)m * 4);
		B[0] = 0;
		C[0] = 0;
		costs[0] = 0;
		if (nl > 0)
			memset(B + 1, 0xff, nl * 4);
		/* the line-breaking DP over the words is not yet written */
		cost = fmtb_cost(w, width, avail, lim, C, nl, nofill);
		nbrk = fmta_break(B, nl, brk, lim);
		free(costs);
		free(C);
		free(B);
		if (cost < 0)	/* placeholder: the DP compares costs here */
			return 0;
	}
	(void)a74;
	(void)a78;
	(void)nofill;
	(void)flag;

	if (fmt_confchanged(f))
		fmt_confupdate(f);
	if (force)
		cost = fmtb_cost(w, width, avail, lim, C, nl, nofill);

	cost = 0;
	nl = 0;
	for (i = 0; i < f->nwords; i++) {
		c = FMT_COST(width, avail, i, lim);
		if (c < cost || cost == 0) {
			cost = c;
			nl = i;
		}
	}
	if (nl > 0 && fmt_confchanged(f))
		return fmt_extractline(f, nl, force) ? 1 : 0;
	return nl > 0 ? 1 : 0;
}
