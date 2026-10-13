#include "op_rr.h"

static void mov_r2r(int a, int b, int t)
{
	int x;

	if (a == b && (t & 0xf) == 8)
		return;
	if (t == 4) {
		op_rr(0x8b, a, b, 4);
		return;
	}
	x = t & 0xf;
	if (x == 4) {
		op_rr((t & 0x10) ? 0x63 : 0x8b, a, b, 8);
	} else if (x == 2) {
		op_rr((t & 0x10) ? 0x10fbf : 0x10fb7, a, b, 8);
	} else if (x == 1) {
		op_rr((t & 0x10) ? 0x10fbe : 0x10fb6, a, b, 8);
	} else {
		op_rr(0x8b, a, b, 8);
	}
}
