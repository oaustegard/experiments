#include "sstr.h"

int
sstr_next(void)
{
	unsigned char c = *sstr_s;

	if (c == 0)
		return -1;
	sstr_s++;
	return c;
}
