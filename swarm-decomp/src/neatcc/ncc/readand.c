#include <string.h>
#include "fn_o_mklocal.h"
#include "fn_o_jz.h"
#include "fn_o_jmp.h"
#include "fn_o_label.h"
#include "fn_o_num.h"
#include "fn_o_local.h"
#include "fn_o_tmpdrop.h"
#include "o_tmpswap.h"
#include "o_assign.h"
#include "o_deref.h"
#include "fn_o_rmlocal.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "ts_de.h"
#include "ts_binop.h"
#include "readxor.h"
#include "readand.h"

static void readand(void)
{
	long tmp;
	long lend;
	int lfalse;
	int l;
	char *p;

	p = tok_see();
	if (p[0] != '&' || p[1] != '&' || p[2])
		return;
	tmp = o_mklocal(4);
	l = label;
	lfalse = l + 1;
	l += 2;
	label = l;
	lend = l;
	ts_de(1);
	nts--;
	o_jz(lend);
	for (;;) {
		p = tok_see();
		if (*p != '&')
			break;
		if (p[1] != '&' || p[2])
			break;
		tok_get();
		readxor();
		p = tok_see();
		while (strcmp("|", p) == 0) {
			if (p[1])
				break;
			tok_get();
			readxor();
			ts_binop(0x13);
			p = tok_see();
		}
		ts_de(1);
		nts--;
		o_jz(lend);
	}
	o_num(1);
	o_local(tmp);
	o_tmpswap();
	o_assign(4);
	o_tmpdrop(1);
	o_jmp(lfalse);
	o_label(lend);
	o_num(0);
	o_local(tmp);
	o_tmpswap();
	o_assign(4);
	o_tmpdrop(1);
	o_label(lfalse);
	o_local(tmp);
	o_deref(0x14);
	o_rmlocal(tmp, 0x14);
	ts[nts].kind = 0x14;
	ts[nts].flags = 0;
	ts[nts].ptr = 0;
	ts[nts].tag = 0;
	nts++;
}
