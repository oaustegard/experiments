#include "ren_field.h"
#include "ren_until.h"
#include "fn_env_wb.h"
#include "fn_env_fmt.h"
#include "fn_fmt_wid.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_tab_next.h"
#include "fn_wb_cat.h"
#include "fn_wb_done.h"
#include "fn_wb_init.h"
#include "fn_wb_wid.h"
#include "wb_hmov.h"

static void ren_field(struct wb *w, int (*next)(void), void (*back)(int))
{
	struct wb fw[32];
	int n, j, wid, tab, sum, k, nf, rem, pos;
	struct wb *fp;

	n = 0;
	goto body;
	while (n != 32) {
body:
		wb_init(&fw[n]);
		n++;
		if (ren_until(&fw[n - 1], next, back, c_fb, c_fa) != 1)
			break;
	}
	nf = (n < 2 ? 2 : n) - 1;
	if (w != env_wb()) {
		wid = wb_wid(w);
	} else {
		wid = fmt_wid(env_fmt());
		wid += wb_wid(env_wb());
		wid -= *nreg(map(".b0"));
	}
	tab = tab_next(wid);
	sum = 0;
	for (k = 0; k < n; k++)
		sum += wb_wid(&fw[k]);
	rem = (tab - wid - sum) % nf;
	nf = (tab - wid - sum) / nf;
	fp = fw;
	pos = wid;
	for (j = 0; ; ) {
		j++;
		wb_hmov(w, pos - wb_wid(w));
		wb_cat(w, fp);
		wb_done(fp);
		if (n == j)
			return;
		fp = &fw[j];
		if (j == n - 1) {
			pos = tab - wb_wid(fp);
			continue;
		}
		pos = wb_wid(w);
		pos += nf;
		pos += (j + rem >= n);
	}
}
