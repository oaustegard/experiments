#include <string.h>
#include "mkfn_proto.h"


#define U16(p) ({ unsigned short w_ = *(const unsigned short *)(p); (int)(unsigned short)((w_ << 8) | (w_ >> 8)); })

/* value of the offset whose bytes start at t[k], osz bytes wide */
#define CFF_VAL(k) \
	({ \
		int v_ = 0, i_; \
		for (i_ = 0; i_ < osz; i_++) \
			v_ = (v_ << 8) + t[(k) + i_]; \
		v_; \
	})

static void
cff_char(const unsigned char *t, int gid, char *name)
{
	int n, a, b, len, osz, o;
	const unsigned char *src;

	if (gid <= 390) {
		strcpy(name, stdset[gid]);
		return;
	}
	n = gid - 390;
	osz = t[2];
	o = n * osz;
	a = CFF_VAL(3 + o);
	b = CFF_VAL(3 + o - osz);
	len = a - b;
	if (mkfn_warn && len > 63)
		fwrite("neatmkfn: truncating glyph names (GNLEN is too small)\n", 1, 54, stderr);
	if (len > 63)
		len = 63;
	src = t + 2 + (U16(t) + 1) * osz + CFF_VAL(3 + o - osz);
	memcpy(name, src, len);
	name[len] = 0;
}
