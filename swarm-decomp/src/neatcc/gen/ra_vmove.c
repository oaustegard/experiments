#include "fn_i_ins.h"
#include "tmpregs.h"

static void ra_vmove(long v, long mask)
{
	int i;
	int *p;
	long r;
	long t;
	long cand;
	long bits;

	r = -1;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] == (int)v) {
			r = i;
			break;
		}
	cand = (func_regs | 0xfc7) & mask;
	bits = 0;
	for (i = 0; i < 16; i++)
		if (ra_lmap[i] >= 0)
			bits |= 1 << i;
	cand &= ~bits;
	bits = 0;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] >= 0)
			bits |= 1 << i;
	cand &= ~bits;
	for (p = tmpregs; p != tmpregs + 14; p++)
		if (cand & (1 << *p)) {
			t = *p;
			if (t >= 0 && ra_vmap[t] < 0 && ra_lmap[t] < 0) {
				i_ins(0x8000400, t, r, 0, 0);
				ra_vmap[t] = v;
				ra_vmap[r] = -1;
				return;
			}
			break;
		}
	val_tomem(v, r);
	ra_vmap[r] = -1;
}
