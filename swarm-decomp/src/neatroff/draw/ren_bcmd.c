#include "wb.h"
#include "fn_wb_init.h"
#include "fn_wb_wid.h"
#include "fn_wb_hmov.h"
#include "fn_wb_vmov.h"
#include "fn_wb_cat.h"
#include "fn_wb_done.h"
#include "fn_sstr_push.h"
#include "fn_sstr_pop.h"
#include "fn_sstr_next.h"
#include "fn_sstr_back.h"
#include "fn_ren_char.h"
#include "fn_nreg.h"
#include "var_dev_res.h"

void ren_bcmd(struct wb *w, char *s)
{
	struct wb b;
	int c, n, maxw, h;

	sstr_push(s);
	wb_init(&b);
	c = sstr_next();
	n = 0;
	maxw = 0;
	while (c >= 0) {
		((void (*)(int))sstr_back)(c);
		ren_char(&b, sstr_next, sstr_back);
		if (wb_wid(&b) > maxw)
			maxw = wb_wid(&b);
		n++;
		wb_hmov(&b, -wb_wid(&b));
		wb_vmov(&b, *nreg('s') * dev_res / 72);
		c = sstr_next();
	}
	sstr_pop();
	h = -(*nreg('s') * dev_res / 72);
	h -= n * (*nreg('s') * dev_res / 72);
	wb_vmov(w, *nreg('s') * dev_res / 72 + h / 2);
	wb_cat(w, &b);
	wb_done(&b);
	wb_vmov(w, h / 2);
	wb_hmov(w, maxw);
}
