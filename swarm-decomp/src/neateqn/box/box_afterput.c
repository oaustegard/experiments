#include "box.h"

static void box_afterput(struct box *b, int c)
{
	unsigned f = b->flags;
	int lo = (unsigned char)f;
	int ch = (unsigned char)c;
	int bit = c & 0x100;
	unsigned v = ch;

	b->nitems++;
	if (ch == '0') {
		if (lo <= 0x70) {
			if ((f & 0xc0) == 0) {
				if ((f & 0xdf) == 0)
					v = 0x10;
			} else if ((0x1000000010001ULL >> (lo - 0x40)) & 1) {
				v = 0x10;
			}
		}
	}
	if (ch == 0x40 && (lo == 0 || lo == 0x50))
		v = 0x10;
	v |= bit;
	b->flags = v;
	if (b->first == 0)
		b->first = v;
}
