#include "fn_tab_type.h"
#include "fn_tab_next.h"
#include "fn_env_wb.h"
#include "fn_env_fmt.h"
#include "fn_fmt_wid.h"
#include "fn_ren_hline.h"
#include "fn_charnext.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_wb_wid.h"

static void ren_tab(struct wb *w, char *s, int (*next)(void), void (*back)(int))
{
	struct wb tw;
	char buf[128];
	int wid;
	int pos;
	int type;
	int c;

	if (w == env_wb())
		wid = fmt_wid(env_fmt()) + wb_wid(env_wb()) - *nreg(map(".b0"));
	else
		wid = wb_wid(w);
	pos = tab_next(wid);
	type = tab_type(wid);
	wb_init(&tw);
	if (type == 'R' || type == 'C') {
		for (;;) {
			c = next();
			if (c == '\t' || c == '\n')
				break;
			if (c == 1 || c < 0)
				break;
			back(c);
			if (charnext(buf, next, back) < 0)
				continue;
			ren_put(&tw, buf, next, back);
		}
		back(c);
		if (type == 'C')
			pos -= wb_wid(&tw) / 2;
		else
			pos -= wb_wid(&tw);
	}
	if (*s != 0 && pos > wid)
		ren_hline(w, pos - wid, s);
	else
		wb_hmov(w, pos - wid);
	wb_cat(w, &tw);
	wb_done(&tw);
}
