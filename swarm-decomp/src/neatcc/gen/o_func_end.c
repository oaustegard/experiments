#include <stdlib.h>
#include "fn_i_ins.h"
#include "fn_ic_get.h"
#include "fn_reg_init.h"
#include "fn_reg_rmap.h"
#include "fn_ic_regcnt.h"
#include "fn_ic_lastuse.h"
#include "fn_reg_mask.h"
#include "fn_i_label.h"
#include "fn_i_reg.h"
#include "fn_reg_safe.h"
#include "fn_reg_lmap.h"
#include "argregs.h"
#include "ic_ent.h"

void o_func_end(void)
{
	long i, j;
	long a1, a2, a3, a4, a5;
	int n;

	ic_get(&ic, &ic_n);
	reg_init(ic, ic_n);
	ic_bbeg = calloc(ic_n, 8);
	ra_gmask = calloc(ic_n, 8);
	loc_mem = calloc(loc_n, 4);
	for (i = 0; i < ic_n; i++) {
		if (ic[i].op & 0xf000)
			ic_bbeg[i] = 1;
		if ((ic[i].op & 0x7000) && ic[i].c < ic_n)
			ic_bbeg[ic[i].c] = 1;
	}
	for (i = 0; i < ic_n; i++) {
		n = ic_regcnt(&ic[i]);
		i_reg((int)ic[i].op, &a1, &a2, &a3, &a4, &a5);
		if (n > 0)
			ra_gmask[ic[i].a] = a2;
		if (n > 1)
			ra_gmask[ic[i].b] = a3;
		if (n > 2)
			ra_gmask[ic[i].c] = a4;
		if ((ic[i].op & 0xffffff) == 0x220000)
			continue;
		if (!(ic[i].op & 0x200))
			continue;
		if (ic[i].c <= 0)
			continue;
		for (j = 0; j < 6 && j < ic[i].c; j++)
			ra_gmask[((long *)ic[i].d)[j]] = 1 << argregs[j];
	}
	for (i = 0; i < 16; i++)
		ra_vmap[i] = -1;
	for (i = 0; i < 16; i++)
		ra_lmap[i] = reg_rmap(0, i);
	func_regs |= (int)reg_mask();
	for (i = 0; i < 64; i++)
		ra_live[i] = -1;
	ra_vmax = 0;
	func_maxargs = 0;
	ic_luse = ic_lastuse(ic, ic_n);
	for (i = 0; i < ic_n; i++) {
		long op = ic[i].op;
		int n0 = ic_regcnt(&ic[i]);
		struct ic_ent *q;
		int n2;

		ic_i = i;
		i_label(i);
		q = &ic[ic_i];
		n2 = ic_regcnt(q);
		op = q->op;
		a1 = 0;
		if ((op & 0xffffff) == 0x210000)
			continue;
		if (n0 < 0 || n2 < 0)
			continue;
		for (j = 0; j < 64; j++) {
			if (ra_live[j] >= 0 && ic_luse[ra_live[j]] <= i)
				ra_vdrop(ra_live[j]);
		}
		if (ic_bbeg[i])
			ra_bbend();
		ra_vdrop(ic[i].a);
		ra_vdrop(ic[i].b);
		ra_spill(ic[i].c);
		ra_spill(ic[i].b);
		ra_vload(ic[i].a, ic[i].b);
		ra_vload(ic[i].a, ic[i].c);
		ra_vload(ic[i].b, ic[i].c);
		ra_vload(ic[i].c, ic[i].a);
		ra_vload(ic[i].c, ic[i].b);
		ra_vmove(ic[i].a, ic[i].c);
		ra_vmove(ic[i].b, ic[i].c);
		if (loc_isread(ic[i].c))
			ra_lmap[(int)ic[i].b] = ic[i].c;
		if (ra_lmap[(int)ic[i].a] < 0 && reg_safe(ic[i].b) && loc_isread(ic[i].b))
			ra_lmap[(int)ic[i].a] = ic[i].b;
		ra_bbend();
		ra_bbend();
	}
	for (i = 0; i < 16; i++) {
		int r = ra_lmap[i];
		if (r >= 0 && r < func_argc && (r > 5 || argregs[r] != i)) {
			loc_mem[r]++;
			i_ins(0x8110000, i, 5, -loc_off[r], 0);
		}
	}
}
