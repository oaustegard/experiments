#include <string.h>
#include <string.h>
#include "fn_tok_see.h"
#include "fn_tok_get.h"
#include "o_bop.h"
#include "o_tmpswap.h"
#include "ts_de.h"
#include "readcmp.h"
#include "readeq.h"
#include "ncc_data.h"

static void readeq(void)
{
	char *p;
	int op;

	readcmp();
	for (;;) {
		p = tok_see();
		if (strcmp("==", p) == 0) {
			tok_get();
			readcmp();
			ts_de(1);
			nts--;
			o_tmpswap();
			ts_de(1);
			nts--;
			o_tmpswap();
			op = 0x82;
		} else {
			p = tok_see();
			if (strcmp(p, "!=") != 0)
				break;
			tok_get();
			readcmp();
			ts_de(1);
			nts--;
			o_tmpswap();
			ts_de(1);
			nts--;
			o_tmpswap();
			op = 0x83;
		}
		o_bop(op);
		ts[nts].flags = 0;
		ts[nts].ptr = 0;
		ts[nts].tag = 0;
		ts[nts++].kind = 0x14;
	}
}
