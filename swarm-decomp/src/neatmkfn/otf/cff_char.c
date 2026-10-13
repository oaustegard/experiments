#include <string.h>
#include "mkfn_proto.h"


#define U16(p) (((p)[0] << 8) | (p)[1])

static void
cff_char(const unsigned char *t, int gid, char *name)
{
	int cnt, osz, n, a, b, len, i;
	const unsigned char *src;

	if (gid <= 390) {
		strcpy(name, stdset[gid]);
		return;
	}
	n = gid - 390;
	cnt = U16(t);
	osz = t[2];
	a = 0;
	for (i = 0; i < osz; i++)
		a = (a << 8) + t[3 + n * osz + i];
	b = 0;
	for (i = 0; i < osz; i++)
		b = (b << 8) + t[3 + (n - 1) * osz + i];
	len = a - b;
	if (len > 63) {
		if (mkfn_warn)
			fwrite("neatmkfn: truncating glyph names (GNLEN is too small)\n", 1, 54, stderr);
		len = 63;
	}
	src = t + 2 + (cnt + 1) * osz + b;
	memcpy(name, src, len);
	name[len] = 0;
}
