#include "o_bop.h"
#include "o_tmpcopy.h"
#include "o_tmpswap.h"
#include "o_assign.h"
#include "ts_de.h"
#include "type_szde.h"
#include "inc_post.h"
#include "ncc_data.h"

static void inc_post(int op)
{
	struct type t;

	t = ts[nts - 1];
	o_tmpcopy();
	ts_de(1);
	o_tmpswap();
	o_tmpcopy();
	ts[nts++] = t;
	ts_de(1);
	t = ts[--nts];
	if (t.ptr > 0) {
		o_num((unsigned)type_szde(&t));
		o_bop(op);
	} else {
		o_num(1);
		o_bop(op);
	}
	o_assign(t.ptr ? 8 : t.kind);
	o_tmpdrop(1);
}
