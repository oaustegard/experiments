#include "fn_i_ins.h"
#include "fn_die.h"

static void ra_vload(long v, int reg)
{
	int i;
	long r;
	int k;
	long x;
	long t;

	if (ra_vmap[reg] == v)
		return;
	if (ra_vmap[reg] >= 0 || ra_lmap[reg] >= 0)
		ra_spill(reg);
	x = (int)v;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] == x)
			break;
	if (i == 16) {
		for (k = 0; k < 64; k++) {
			t = ra_live[k];
			if (v == t)
				break;
		}
		if (k == 64) {
			die("neatcc: the specified value is not live\n");
			k = 0;
		}
		i_ins(0x8110000, reg, 5, -8 - (loc_pos + 8 * (long)k), 0);
		ra_vmap[reg] = v;
		return;
	} else {
		r = -1;
		for (i = 0; i < 16; i++)
			if (ra_vmap[i] == x) {
				r = i;
				break;
			}
		i_ins(0x8000400, reg, r, 0, 0);
		for (i = 0; i < 16; i++)
			if (ra_vmap[i] == x)
				break;
		if (i == 16)
			i = -1;
		ra_vmap[i] = -1;
		ra_vmap[reg] = v;
	}
}
