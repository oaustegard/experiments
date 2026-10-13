#include "wb.h"
#include "fn_wb_hpos.h"
#include "fn_wb_vpos.h"
#include "fn_wb_hmov.h"
#include "fn_wb_vmov.h"
#include "fn_sstr_push.h"
#include "fn_sstr_pop.h"
#include "fn_sstr_next.h"
#include "fn_sstr_back.h"
#include "fn_ren_char.h"

void ren_zcmd(struct wb *w, char *s)
{
	int c, hpos, vpos;

	hpos = wb_hpos(w);
	vpos = wb_vpos(w);
	sstr_push(s);
	while ((c = sstr_next()) >= 0) {
		((void (*)(int))sstr_back)(c);
		ren_char(w, sstr_next, sstr_back);
	}
	sstr_pop();
	wb_hmov(w, hpos - wb_hpos(w));
	wb_vmov(w, vpos - wb_vpos(w));
}
