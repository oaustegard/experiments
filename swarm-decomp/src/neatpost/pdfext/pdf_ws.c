#include <ctype.h>
#include "sys_std.h"

int pdf_ws(unsigned char *s, int end, int i)
{
	int j = i;

	if (i >= end)
		return 0;
	while (isspace(s[j])) {
		j++;
		if (j >= end)
			return end - i;
	}
	return j - i;
}
