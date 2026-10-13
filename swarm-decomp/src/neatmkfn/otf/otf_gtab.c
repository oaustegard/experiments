#include <stdlib.h>
#include <string.h>

#define U16(p) (((p)[0] << 8) | (p)[1])

static int
otf_gtab(const unsigned char *t, struct otf_feat *ents, int unused)
{
	const unsigned char *sl, *r, *sp, *lr;
	char stag[8], ltag[8];
	int cnt, nscr, nlang, i, j;

	sl = t + U16(t + 4);
	nscr = U16(sl);
	cnt = 0;
	for (i = 0; i < nscr; i++) {
		r = sl + 2 + 6 * i;
		memcpy(stag, r, 4);
		stag[4] = 0;
		if (!mkfn_script(stag, nscr))
			continue;
		sp = sl + U16(r + 4);
		nlang = U16(sp + 2);
		if (U16(sp) != 0) {
			if (mkfn_lang(0, nlang + 1))
				cnt = otf_lang(t, sp + U16(sp), stag, "", ents, cnt, sp[1]);
		}
		if (nlang == 0)
			continue;
		for (j = 0; j < nlang; j++) {
			lr = sp + 4 + 6 * j;
			memcpy(ltag, lr, 4);
			ltag[4] = 0;
			if (mkfn_lang(ltag, nlang + (U16(sp) != 0)))
				cnt = otf_lang(t, sp + U16(lr + 4), stag, ltag, ents, cnt, lr[0]);
		}
	}
	qsort(ents, cnt, sizeof *ents, lookupcmp);
	return cnt;
}
