#include <ctype.h>
#include "sys_std.h"
#include "fn_pdf_len.h"

int pdf_dkey(unsigned char *s, int end, int i, int n)
{
	int k = 0, pos = i + 2;

	if (end <= i + 4)
		return -1;
	do {
		if (s[pos] == '>' && s[pos + 1] == '>')
			return -1;
		while (pos < end && isspace(s[pos]))
			pos++;
		if (k == n)
			return pos;
		if (pos < end)
			pos += pdf_len(s, end, pos);
		if (pos < end)
			pos += pdf_len(s, end, pos);
		while (pos < end && isspace(s[pos]))
			pos++;
		k++;
	} while (pos + 2 < end);
	return -1;
}
