#include "fmt.h"
#include "fmt_fillwords.h"
#include "fmtb_cost.h"
#include "fn_map.h"
#include "fn_nreg.h"

/* line-breaking cost of the break list brk[0..n-1] over the words w;
   hyc[] holds the hyphenation penalties of the .hycost registers */
static long fmtb_cost(struct fmtword *w, int a, int b, int c, int *brk, int n)
{
	long total, r;
	int hyc[3];
	int i, k, prev, nhy, width, d1, d2, pmll, t;

	hyc[0] = *nreg(map(".hycost"));
	hyc[1] = *nreg(map(".hycost")) + *nreg(map(".hycost2"));
	hyc[2] = *nreg(map(".hycost3")) + hyc[1];
	total = 0;
	prev = 0;
	nhy = 0;
	for (i = 0; i < n; i++) {
		k = brk[i];
		width = w[k - 1].pos - w[prev].pos + w[prev].wid;
		d1 = w[k - 1].f2c - w[prev].f2c;
		d2 = w[k - 1].f30 - w[prev].f30;
		if (b && i == n - 1) {
			pmll = *nreg(map(".pmll"));
			if (pmll != 0) {
				t = a * pmll / 100;
				if (width < t)
					total += (long)*nreg(map(".pmllcost")) * (t - width) / t;
			}
			prev = k;
			continue;
		}
		if (d1 == 0)
			d1 = 1;
		r = w[k - 1].f20 + FMT_COST(a, width, d1, d2);
		if (w[k - 1].f18 == 0) {
			total += r;
			prev = k;
			nhy = 0;
			continue;
		}
		if (c == i + 1)
			r += 10000000;
		nhy++;
		if (*nreg(map(".hlm")) > 0 && nhy >= *nreg(map(".hlm"))) {
			total += r + 10000000;
			prev = k;
			continue;
		}
		if (nhy <= 1)
			total += r + hyc[1];
		else
			total += r + hyc[2];
		prev = k;
	}
	return total;
}
