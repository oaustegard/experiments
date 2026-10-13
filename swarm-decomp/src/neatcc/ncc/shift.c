#include "type.h"
#include "ncc_data.h"
#include "readadd.h"
#include "ts_de.h"
#include "o_bop.h"
#include "o_tmpswap.h"
#include "shift.h"

/* << and >>: the right operand is read here; the left one is already on the type stack */
static void shift(int op)
{
	int ptr, kind;
	unsigned sz;

	readadd();
	ts_de(1);
	nts--;
	o_tmpswap();
	ts_de(1);
	nts--;
	ptr = ts[nts].ptr;
	kind = ts[nts].kind;
	o_tmpswap();
	if (ptr) {
		o_bop(op | 0x8000000);
		sz = 8;
	} else {
		o_bop((unsigned)kind << 24 | op);
		sz = kind & 0xf;
		sz = sz < 4 ? 4 : sz;
		if (kind & 0x10) {
			sz |= 0x10;
		} else {
			kind &= 0xc;
			if (kind == 0)
				sz = 0x14;
		}
	}
	ts[nts].flags = 0;
	ts[nts].ptr = 0;
	ts[nts].tag = 0;
	nts++;
	ts[nts - 1].kind = sz;
}
