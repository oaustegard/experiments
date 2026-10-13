#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_box_puttext.h"
#include "fn_nreg.h"
#include "e_vars.h"

static int eqn_gaps(struct box *b, int ps)
{
	if (tok_jmp("~") == 0) {
		box_puttext(b, 0x14, "\\h'%du*%sp/100u'", e_thickspace, nreg(ps));
		return 0;
	}
	if (tok_jmp("^") == 0) {
		box_puttext(b, 0x14, "\\h'%du*%sp/100u'", e_thinspace, nreg(ps));
		return 0;
	}
	if (tok_jmp("\t") == 0) {
		box_puttext(b, 0x14, "\t");
		return 0;
	}
	return 1;
}
