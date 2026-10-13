#include "ren_until.h"
#include "ren_page.h"
#include "ren_line.h"
#include "fn_map.h"
#include "fn_num_str.h"
#include "fn_nreg.h"
#include "fn_charnext.h"
#include "fn_sstr_back.h"
#include "fn_sstr_next.h"
#include "fn_wb_buf.h"
#include "fn_wb_cat.h"
#include "fn_wb_done.h"
#include "fn_wb_init.h"
#include "fn_wb_wid.h"
#include "sstr_push.h"
#include "sstr_pop.h"
#include "var_tr_c.h"
#include "wb_hmov.h"

void ren_tl(int (*get)(void), void (*back)(int))
{
	struct wb a, b;
	char buf[32];
	char *s;
	int lt, wd;

	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	s = num_str(map("%"));
	wb_init(&a);
	wb_init(&b);
	charnext(buf, get, back);
	if (strcmp(buf, "\n") == 0)
		back('\n');
	while (ren_until(&b, get, back, c_pc, buf) == 1) {
		sstr_push(s);
		ren_until(&b, sstr_next, (void (*)(int))sstr_back, buf, NULL);
		sstr_pop();
	}
	wb_hmov(&a, -wb_wid(&a));
	wb_cat(&a, &b);
	while (ren_until(&b, get, back, c_pc, buf) == 1) {
		sstr_push(s);
		ren_until(&b, sstr_next, (void (*)(int))sstr_back, buf, NULL);
		sstr_pop();
	}
	lt = *nreg(map(".lt"));
	wd = (lt - wb_wid(&b)) / 2 - wb_wid(&a);
	wb_hmov(&a, wd);
	wb_cat(&a, &b);
	while (ren_until(&b, get, back, c_pc, buf) == 1) {
		sstr_push(s);
		ren_until(&b, sstr_next, (void (*)(int))sstr_back, buf, NULL);
		sstr_pop();
	}
	lt = *nreg(map(".lt"));
	wb_hmov(&a, lt - wb_wid(&b) - wb_wid(&a));
	wb_cat(&a, &b);
	ren_line(wb_buf(&a), wb_wid(&a), 2, 0, 0, 0, *nreg(map(".lt")), a.a38[0], a.a38[1]);
	wb_done(&b);
	wb_done(&a);
}
