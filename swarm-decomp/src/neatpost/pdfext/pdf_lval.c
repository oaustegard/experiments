#include <ctype.h>
#include "sys_std.h"
#include "fn_pdf_len.h"

int pdf_lval(unsigned char *s, int end, int i, int want)
{
	int pos = i + 1;
	int n = 0;

	while (pos < end) {
		if (s[pos] == ']')
			return -1;
		if (n == want)
			return pos;
		pos += pdf_len(s, end, pos);
		if (end <= pos)
			return -1;
		while (pos < end && isspace(s[pos]))
			pos++;
		n++;
	}
	return -1;
}
