#include "op_rr.h"

static void mov_r2r(int a, int b, int t)
{
	int x;
	int op;

	if (a == b && (t & 0xf) == 8)
		return;
	if (t == 4) {
		op_rr(0x8b, a, b, 4);
		return;
	}
	x = t & 0xf;
	if (x == 4) {
		op = (t & 0x10) ? 0x63 : 0x8b;
		goto common;
	}
	if (x == 2) {
		op = (t & 0x10) ? 0x10fbf : 0x10fb7;
		goto common;
	}
	op = 0x8b;
	if (x == 1)
		op = (t & 0x10) ? 0x10fbe : 0x10fb6;
common:
	op_rr(op, a, b, 8);
}
