#include "fn_die.h"
#include "tmpregs.h"

static long ra_regget(int v, long pref, long want, long avoid)
{
	int i;
	int j;
	int k;
	int *p;
	int *q;
	long allow;
	long vb;
	long lb;
	long cand;

	allow = want & ~avoid;
	pref &= allow;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] == v)
			break;
	if (i < 16) {
		for (j = 0; j < 16; j++)
			if (ra_vmap[j] == v)
				break;
		if (j == 16)
			j = -1;
		if ((1 << j) & allow) {
			for (k = 0; k < 16; k++)
				if (ra_vmap[k] == v)
					break;
			return k;
		}
	}
	vb = 0;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] >= 0)
			vb |= 1 << i;
	lb = 0;
	for (i = 0; i < 16; i++)
		if (ra_lmap[i] >= 0)
			lb |= 1 << i;
	vb = ~vb;
	lb = ~lb;
	cand = pref & vb & lb;
	for (p = tmpregs; p != tmpregs + 14; p++)
		if (cand & (1 << *p)) {
			if (*p < 0)
				break;
			for (q = tmpregs; q != tmpregs + 14; q++)
				if (cand & (1 << *q))
					return *q;
			return -1;
		}
	cand = allow & vb & lb;
	for (p = tmpregs; p != tmpregs + 14; p++)
		if (cand & (1 << *p)) {
			if (*p < 0)
				break;
			for (q = tmpregs; q != tmpregs + 14; q++)
				if (cand & (1 << *q))
					return *q;
			return -1;
		}
	for (p = tmpregs; p != tmpregs + 14; p++)
		if (pref & (1 << *p)) {
			if (*p < 0)
				break;
			for (q = tmpregs; q != tmpregs + 14; q++)
				if (pref & (1 << *q))
					return *q;
			return -1;
		}
	for (p = tmpregs; p != tmpregs + 14; p++)
		if (allow & (1 << *p)) {
			if (*p < 0)
				break;
			for (q = tmpregs; q != tmpregs + 14; q++)
				if (allow & (1 << *q))
					return *q;
			return -1;
		}
	die("neatcc: cannot allocate an acceptable register\n");
	return 0;
}
