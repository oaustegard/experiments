#include <string.h>
#include "otf_feat.h"

#define U16(p) (((p)[0] << 8) | (p)[1])

static int
otf_featrec(const unsigned char *t, const unsigned char *rec, const char *name,
	const char *str, struct otf_feat *ents, int n, int unused)
{
	char tag[8] = {0};
	const unsigned char *fe, *lp, *end;
	int k, idx;
	unsigned int cnt;

	memcpy(tag, rec, 4);
	fe = t + U16(t + 6) + U16(rec + 4);
	cnt = U16(fe + 2);
	if (cnt == 0)
		return n;
	end = fe + 6 + 2 * (size_t)(cnt - 1);
	lp = fe + 4;
	do {
		idx = U16(lp);
		for (k = 0; k < n; k++) {
			if (ents[k].idx == idx && ents[k].str[0] == 0 &&
			    strcmp(ents[k].name, name) == 0 &&
			    strcmp(ents[k].tag, tag) == 0)
				break;
		}
		if (k == n) {
			n++;
			strcpy(ents[k].tag, tag);
			strcpy(ents[k].name, name);
			strcpy(ents[k].str, str);
			ents[k].idx = U16(lp);
		}
		lp += 2;
	} while (lp != end);
	return n;
}
