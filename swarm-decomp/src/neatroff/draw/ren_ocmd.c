#include "wb.h"
#include "fn_wb_init.h"
#include "fn_wb_wid.h"
#include "fn_wb_hmov.h"
#include "fn_wb_cat.h"
#include "fn_wb_done.h"
#include "fn_sstr_push.h"
#include "fn_sstr_pop.h"
#include "fn_sstr_next.h"
#include "fn_sstr_back.h"
#include "fn_ren_char.h"

void ren_ocmd(struct wb *w, char *s)
{
	struct wb b, t;
	int c, n, maxw, wid;

	sstr_push(s);
	wb_init(&b);
	wb_init(&t);
	c = sstr_next();
	maxw = 0;
	while (c >= 0) {
		((void (*)(int))sstr_back)(c);
		ren_char(&t, sstr_next, sstr_back);
		wid = wb_wid(&t);
		if (wid > maxw)
			maxw = wid;
		wid = -(wid / 2);
		wb_hmov(&b, wid);
		wb_cat(&b, &t);
		wb_hmov(&b, wid);
		c = sstr_next();
	}
	n = maxw / 2;
	sstr_pop();
	wb_hmov(w, n);
	wb_cat(w, &b);
	wb_hmov(w, n);
	wb_done(&t);
	wb_done(&b);
}
