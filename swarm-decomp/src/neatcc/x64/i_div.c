#include "op_rr.h"
#include "op_x.h"

static void i_div(int op, long unused, int c)
{
	int s = (op >> 24) & 0x10;

	if (c != 2) {
		if (!s)
			op_rr(0x33, 2, 2, 4);
		else
			op_x(0x99, 0, 2, 8);
	}
	op_rr(0xf7, s ? 7 : 6, c, 8);
}
