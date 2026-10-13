#include "fn_o_num.h"
#include "o_bop.h"
#include "o_tmpswap.h"
#include "ts_de.h"
#include "bt_op.h"
#include "type_szde.h"
#include "ts_addop.h"

static void ts_addop(int op)
{
	struct type t1, t2;
	struct type *t;
	int sz;

	ts_de(1);
	nts--;
	t1 = ts[nts];
	o_tmpswap();
	ts_de(1);
	nts--;
	t2 = ts[nts];
	o_tmpswap();
	if (!t1.ptr) {
		if (!t2.ptr) {
			o_bop(op);
			{
				struct type r;
				r.kind = bt_op(t1.kind, t2.kind);
				r.flags = 0;
				r.ptr = 0;
				r.tag = 0;
				ts[nts++] = r;
			}
			return;
		}
		if ((unsigned)type_szde(&t2) > 1) {
			o_num(type_szde(&t2));
			o_bop(0x40);
		}
		o_bop(op);
		t = &t2;
	} else {
		if (t2.ptr) {
			o_bop(op);
			sz = type_szde(&t1);
			if (sz > 1) {
				o_num(sz);
				o_bop(0x41);
			}
			{
				struct type r;
				r.kind = 0x18;
				r.flags = 0;
				r.ptr = 0;
				r.tag = 0;
				ts[nts++] = r;
			}
			return;
		}
		o_tmpswap();
		o_tmpswap();
		o_bop(op);
		t = &t1;
	}
	ts[nts++] = *t;
}
