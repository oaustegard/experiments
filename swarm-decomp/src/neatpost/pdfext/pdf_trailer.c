#include "sys_std.h"

int pdf_trailer(unsigned char *s, int end)
{
	int line = prevline(s, end, 0);
	int k;
	unsigned char c;

	for (;;) {
		k = 0;
		while ((c = s[line + k]) != 0) {
			unsigned char d = (unsigned char)"trailer"[k];
			if (d == 0)
				break;
			k++;
			if (c != d)
				goto nomatch;
		}
		{
			unsigned char *p = (unsigned char *)memchr(s + line, '\n', end - line);
			if (p == 0)
				return -1;
			return p - s + 1;
		}
nomatch:
		line = prevline(s, line, 0);
		if (line < 0)
			return -1;
	}
}
