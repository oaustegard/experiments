#include "utf8_len.h"

static int utf8_len(int c)
{
	if ((c & 0xc0) != 0xc0)
		return c > 0;
	if (!(c & 0x20))
		return 2;
	if (!(c & 0x10))
		return 3;
	if (!(c & 8))
		return 4;
	return 1;
}
