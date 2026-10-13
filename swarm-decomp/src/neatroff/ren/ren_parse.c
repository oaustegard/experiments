#include "ren_put.h"
#include "fn_charnext.h"
#include "fn_odiv_beg.h"
#include "fn_odiv_end.h"
#include "fn_sstr_back.h"
#include "fn_sstr_next.h"
#include "sstr_pop.h"
#include "sstr_push.h"

int ren_parse(struct wb *w, char *t)
{
	char buf[0x80];
	int c;

	odiv_beg();
	sstr_push(t);
	while ((c = sstr_next()) >= 0) {
		((void (*)(int))sstr_back)(c);
		if (charnext(buf, sstr_next, sstr_back) < 0)
			break;
		ren_put(w, buf, sstr_next, sstr_back);
	}
	sstr_pop();
	odiv_end();
	return 0;
}
