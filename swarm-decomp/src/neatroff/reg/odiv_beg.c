#include "env.h"
#include "fn_map.h"

void odiv_beg(void)
{
	int i = nodivs++;
	int r, k;

	r = 102;
	k = eregs_idx[r];
	odivs[i][0] = !k ? nregs[r] : ((int *)env)[k];
	r = 115;
	k = eregs_idx[r];
	odivs[i][1] = !k ? nregs[r] : ((int *)env)[k];
	r = 109;
	k = eregs_idx[r];
	odivs[i][2] = !k ? nregs[r] : ((int *)env)[k];
	r = map(".f0");
	k = eregs_idx[r];
	odivs[i][3] = !k ? nregs[r] : ((int *)env)[k];
	r = map(".s0");
	k = eregs_idx[r];
	odivs[i][4] = !k ? nregs[r] : ((int *)env)[k];
	r = map(".m0");
	k = eregs_idx[r];
	odivs[i][5] = !k ? nregs[r] : ((int *)env)[k];
	r = map(".cd");
	k = eregs_idx[r];
	odivs[i][6] = !k ? nregs[r] : ((int *)env)[k];
}
