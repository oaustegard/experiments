#include "ic_ent.h"
#include "ic_tab.h"
#include "ic_off.h"
#include "fn_i_imm.h"
#include "fn_i_reg.h"

static int io_loc(void)
{
	struct ic_ent *e = &ic[ic_n - 1];
	struct ic_ent *f;
	long x, y, p[5];
	long yy;
	long lr;

	if ((e->op & 0x110000) == 0x110000) {
		int r = ic_off(ic, e->a, &x, &y);
		if (r)
			return r;
		yy = y;
		f = &ic[x];
		if (!(~f->op & 0x200400)) {
			e->op = (e->op & ~0x300000) | 0x200000;
			e->a = f->a;
			e->b += yy + f->b;
			return 0;
		}
		lr = i_reg(e->op | 0x100000, &p[0], &p[1], &p[2], &p[3], &p[4]);
		if (lr)
			return lr;
		if (!i_imm(p[2], yy))
			return 0;
		e->a = x;
		return e->b = y + e->b;
	}
	if ((e->op & 0x120000) == 0x120000) {
		int r = ic_off(ic, e->b, &x, &y);
		if (r)
			return r;
		yy = y;
		f = &ic[x];
		if (!(~f->op & 0x200400)) {
			e->op = (e->op & ~0x300000) | 0x200000;
			e->b = f->a;
			e->c += yy + f->b;
			return 0;
		}
		lr = i_reg(e->op | 0x100000, &p[0], &p[1], &p[2], &p[3], &p[4]);
		if (lr)
			return lr;
		if (!i_imm(p[3], yy))
			return 0;
		e->b = x;
		return e->c = y + e->c;
	}
	return 0;
}
