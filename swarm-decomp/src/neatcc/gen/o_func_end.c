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
#include "fn_i_code.h"
#include "fn_i_wrap.h"
#include "fn_ic_free.h"
#include "fn_reg_done.h"
#include "fn_o_tmpdrop.h"
#include "fn_o_back.h"
#include "fn_die.h"
#include "fn_mem_len.h"
#include "fn_mem_put.h"
#include "fn_out_rel.h"
#include "argregs.h"
#include "ic_ent.h"
#include "mem.h"

void o_func_end(void)
{
	long i, j, k, n, op, opx, mask, sarg;
	long a1, a2, a3, a4, a5;
	struct ic_ent *p;
	long *code_rel[3];
	char *code;
	int n0, n2, r, any;

	ic_get(&ic, &ic_n);
	reg_init(ic, ic_n);
	n = ic_n;
	p = ic;
	ic_bbeg = calloc(n, 8);
	ra_gmask = calloc(n, 8);
	loc_mem = calloc(loc_n, 4);
	for (i = 0; i < n; i++) {
		if (p[i].op & 0xf000)
			ic_bbeg[i] = 1;
		if ((p[i].op & 0x7000) && p[i].c < n)
			ic_bbeg[p[i].c] = 1;
	}
	for (i = 0; i < n; i++) {
		r = ic_regcnt(&p[i]);
		i_reg((int)p[i].op, &a1, &a2, &a3, &a4, &a5);
		if (r > 0)
			ra_gmask[p[i].a] = a2;
		if (r > 1)
			ra_gmask[p[i].b] = a3;
		if (r > 2)
			ra_gmask[p[i].c] = a4;
		op = p[i].op;
		if ((op & 0xffffff) == 0x220000) {
			if (reg_lmap(i, p[i].b))
				ra_gmask[p[i].a] = 1 << reg_lmap(i, p[i].b);
		}
		if (op & 0x200) {
			if (p[i].c > 0) {
				for (j = 0; j < 6 && j < p[i].c; j++)
					ra_gmask[((long *)p[i].d)[j]] = 1 << argregs[j];
			}
		}
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
	for (i = 0; i < 16; i++) {
		r = ra_lmap[i];
		if (r >= 0 && r < func_argc && (r > 5 || argregs[r] != i)) {
			loc_mem[r]++;
			i_ins(0x8110000, i, 5, -loc_off[r], 0);
		}
	}
	for (i = 0; i < ic_n; i++) {
		op = ic[i].op;
		opx = op & 0xffffff;
		n0 = ic_regcnt(&ic[i]);
		ic_i = i;
		i_label(i);
		n2 = ic_regcnt(&ic[ic_i]);
		op = ic[ic_i].op;
		a1 = 0;
		if ((op & 0xffffff) == 0x210000) {
			k = ic[ic_i].a;
			for (j = 0; j < 16; j++)
				if (ra_lmap[j] == k)
					break;
		}
		mask = 0;
		if ((op & 0x200) && ic[ic_i].c > 0) {
			for (j = 0; j < 6 && j < ic[ic_i].c; j++)
				mask |= 1 << argregs[j];
		}
		if (i_reg(ic[ic_i].op, &a1, &a2, &a3, &a4, &a5))
			die("neatcc: instruction %08lx not supported\n", ic[ic_i].op);
		if (ic[ic_i].op & 0x6000) {
			for (k = 0; k < 16; k++) {
				if (reg_rmap(ic_i, k) < 0)
					continue;
				if (ra_lmap[k] == reg_rmap(ic_i, k))
					continue;
				mask |= 1L << k;
			}
		}
		if (n2 > 1) {
			/* two register operands: allocate each (ra_regget) */
			r = ra_regget(ic[ic_i].b, a3, a3, mask);
			mask |= 1 << r;
			k = ra_regget(ic[ic_i].a, a2, a2, mask);
			mask |= 1 << k;
		} else if (n2 == 1) {
		}
		if (op & 0x200) {
			ra_vdrop(ic[i].a);
			ra_vdrop(ic[i].b);
		}
		func_regs |= (int)(a1 | mask);
		if (n0 > 0) {
		}
		if (ic_bbeg[i] && i + 1 < ic_n) {
			if (op & 0x7000)
				ra_bbend();
		}
		if (op & 0x100)
			i_ins(op, 0, 0, 0, 0);
		opx = op & 0xffffff;
		if (opx == 0x110000) {
			i_ins(op, 0, 0, ic[i].b, 0);
		} else if (opx == 0x210000) {
			i_ins(op, 0, 0, 0, 0);
		} else if (opx == 0x120000) {
			i_ins(op, 0, 0, ic[i].c, 0);
		} else if (opx == 0x220000) {
			i_ins(op, 0, 0, ic[i].c, 0);
		} else if (opx == 0x8000) {
			i_ins(op, 0, 0, 0, 0);
		} else if (opx == 0x400) {
			i_ins(op, 0, 0, ic[i].c, 0);
		} else if (opx == 0x1000) {
			i_ins(op, 0, 0, ic[i].c, 0);
		} else if (opx == 0x800 || opx == 0x801) {
			i_ins(op, 0, 0, 0, 0);
		} else if (opx == 0x200) {
			i_ins(op, 0, 0, 0, 0);
		} else {
			i_ins(op, 0, 0, 0, 0);
		}
		if (op & 0x2000)
			i_ins(op, 0, 0, ic[i].c, 0);
		if (op & 0x4000)
			i_ins(op, 0, 0, ic[i].c, 0);
		if (i + 1 < ic_n && ic_bbeg[i + 1])
			ra_bbend();
		for (j = 0; j < 64; j++) {
			if (ra_live[j] >= 0 && ic_luse[ra_live[j]] <= i)
				ra_vdrop(ra_live[j]);
		}
		ra_spill(ic[i].c);
		ra_spill(ic[i].b);
		for (k = 0; k < 16; k++) {
			if (a1 & (1L << k))
				ra_spill((int)k);
		}
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
	}
	i_label(ic_n);
	free(ic_luse);
	k = -1;
	for (i = 0; i < func_argc; i++)
		if (loc_mem[i])
			k = i + 1;
	sarg = 0;
	for (i = 0; i < 6; i++)
		if (func_varg || k > i)
			sarg |= 1 << argregs[i];
	any = 0;
	for (i = 0; i < loc_n; i++)
		if (loc_mem[i])
			any = 1;
	if (any) {
		for (i = 0; i < 14; i++)
			if (tmpregs[i] >= 0)
				mask |= 1 << tmpregs[i];
	}
	i_wrap(func_argc, mask, loc_pos, func_varg, func_regs & 0xf008, 0);
	free(ic_bbeg);
	free(ra_gmask);
	free(loc_mem);
	i_code(&code, &a1, &code_rel[0], &code_rel[1], &code_rel[2], &a5);
	for (j = 0; j < a5; j++)
		out_rel(code_rel[0][j], code_rel[1][j], code_rel[2][j] + mem_len(&cs));
	mem_put(&cs, code, a1);
	free(code);
	free(code_rel[0]);
	free(code_rel[1]);
	free(code_rel[2]);
	for (i = 0; i < ic_n; i++)
		ic_free(&ic[i]);
	free(ic);
	reg_done();
	o_tmpdrop(-1);
	o_back(0);
	free(loc_off);
	loc_off = NULL;
	loc_n = 0;
	loc_sz = 0;
	loc_pos = 0;
}
