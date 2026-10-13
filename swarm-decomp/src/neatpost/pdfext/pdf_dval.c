#include <ctype.h>
#include "sys_std.h"
#include "fn_pdf_len.h"

int pdf_dval(unsigned char *s, int end, int i, const char *key)
{
	int pos = i + 2;
	int len, j, q;

	if (end <= i + 4)
		return -1;
	do {
		if (s[pos] == '>' && s[pos + 1] == '>')
			return -1;
		while (pos < end && isspace(s[pos]))
			pos++;
		len = pdf_len(s, end, pos);
		q = pos + len;
		if (len == strlen(key)) {
			for (j = 0; key[j] != 0; j++) {
				if (s[pos + j] == 0)
					break;
				if (s[pos + j] != key[j])
					goto mismatch;
			}
			pos = q;
			while (pos < end && isspace(s[pos]))
				pos++;
			return pos;
		}
mismatch:
		pdf_len(s, end, pos);
		pos = q;
		if (end > q)
			pos = q + pdf_len(s, end, q);
		while (pos < end && isspace(s[pos]))
			pos++;
	} while (pos + 2 < end);
	return -1;
}
