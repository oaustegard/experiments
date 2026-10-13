#include "region.h"
#include "ic_ent.h"
#include "reg_region.h"

static long reg_region(struct ic_ent *tab, long loc, long pos, long *lo, long *hi, char *seen, long unused)
{
	long cnt = 0, s;

	if (pos < 0)
		return cnt;
	for (;;) {
		if (*lo > pos)
			*lo = pos;
		if (*hi <= pos)
			*hi = pos + 1;
		if (seen[pos])
			return cnt;
		seen[pos] = 1;
		if ((tab[pos].op & 0xffffff) == 0x220000 && tab[pos].b == loc)
			return cnt;
		if ((tab[pos].op & 0xffffff) == 0x210000 && tab[pos].a == loc)
			cnt++;
		for (s = dst_head[pos]; s >= 0; s = dst_next[s])
			cnt += reg_region(tab, loc, s, lo, hi, seen, unused);
		if (pos == 0 || (tab[pos - 1].op & 0x1000))
			return cnt;
		pos--;
	}
}
