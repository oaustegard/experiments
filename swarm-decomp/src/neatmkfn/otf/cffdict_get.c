#include "cff_op.h"

static int
cffdict_get(const unsigned char *p, int len, int key)
{
	int off = 0, v = 0, prev;

	for (;;) {
		if (off >= len)
			return 0;
		prev = v;
		off += cff_op(p, off, &v);
		if (v == 12) {
			off += cff_op(p, off, &v);
			v += 0x4b0;
		}
		if (v == key)
			return prev;
	}
}
