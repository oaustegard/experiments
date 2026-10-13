#include "fn_reg_rmap.h"
#include "fn_i_ins.h"
#include "ic_ent.h"
static void ra_bbend(void)
{
	long reg;
	long l;
	long r;
	int loc;

	for (reg = 0; reg < 16; reg++)
		if (ra_vmap[reg] >= 0)
			ra_spill(reg);
	for (reg = 0; reg < 16; reg++) {
		l = ra_lmap[reg];
		r = reg_rmap(ic_i, reg);
		if (l != r)
			if (ra_lmap[reg] >= 0)
				ra_spill(reg);
	}
	for (reg = 0; reg < 16; reg++) {
		l = ra_lmap[reg];
		r = reg_rmap(ic_i, reg);
		if (l != r) {
			loc = reg_rmap(ic_i, reg);
			ra_lmap[reg] = loc;
			loc_mem[loc]++;
			i_ins(0x8110000, reg, 5, -loc_off[loc], 0);
		}
	}
}
