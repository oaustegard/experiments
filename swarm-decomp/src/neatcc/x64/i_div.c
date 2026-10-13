#include "op_rr.h"
#include "op_x.h"

static void i_div(int op, int c)
{
	int s = (op >> 24) & 0x10;

	if (c == 2) {
		if (s)
			op_rr(0xf7, 7, c, 8);
		else
			op_rr(0xf7, 6, c, 8);
		return;
	}
	if (s) {
		op_x(0x99, 0, 2, 8);
		op_rr(0xf7, 7, c, 8);
	} else {
		op_rr(0x33, 2, 2, 4);
		op_rr(0xf7, 6, c, 8);
	}
}
