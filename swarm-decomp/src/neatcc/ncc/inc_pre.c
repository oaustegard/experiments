#include "type.h"
#include "ncc_data.h"
#include "fn_o_num.h"
#include "o_bop.h"
#include "o_assign.h"
#include "o_tmpcopy.h"
#include "ts_de.h"
#include "type_szde.h"
#include "readpre.h"
#include "inc_pre.h"

/* prefix ++ / --: op is the binary operator, applied to the operand read by readpre */
static void inc_pre(int op)
{
	struct type t;
	struct type *p;

	readpre();
	o_tmpcopy();
	p = &ts[nts - 1];
	nts++;
	p[1] = p[0];
	ts_de(1);
	nts--;
	t = ts[nts];
	if (!t.ptr)
		o_num(1);
	else
		o_num((unsigned)type_szde(&t));
	o_bop(op);
	o_assign(t.ptr ? 8 : t.kind);
	ts_de(0);
}
