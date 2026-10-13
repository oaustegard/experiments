#include <string.h>
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "ts_de.h"
#include "o_tmpswap.h"
#include "o_bop.h"
#include "bt_op.h"
#include "readadd.h"
#include "shift.h"
#include "cmp.h"
#include "ncc_data.h"

static void cmp(int op)
{
	char *p;
	int kl, kr, fl, fr;

	readadd();
	for (;;) {
		p = tok_see();
		if (strcmp("<<", p) == 0) {
			tok_get();
			shift(0x20);
			continue;
		}
		p = tok_see();
		if (strcmp(">>", p) == 0) {
			tok_get();
			shift(0x21);
			continue;
		}
		break;
	}
	ts_de(1);
	nts--;
	kl = ts[nts].kind;
	fl = ts[nts].ptr;
	o_tmpswap();
	ts_de(1);
	nts--;
	fr = ts[nts].ptr;
	kr = ts[nts].kind;
	o_tmpswap();
	if (fr)
		kr = 8;
	if (fl)
		kl = 8;
	o_bop((bt_op(kl, kr) << 24) | op);
	ts[nts].flags = 0;
	ts[nts].ptr = 0;
	ts[nts].tag = 0;
	ts[nts++].kind = 0x14;
}
