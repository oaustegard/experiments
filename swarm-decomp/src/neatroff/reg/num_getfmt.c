#include <string.h>

char *num_getfmt(int r)
{
	static char fmtbuf[128];
	int f = nregs_fmt[r];
	unsigned int w, n;
	char *p = fmtbuf;

	if ((char)f == '0' || (f & 0xdf) == 0x58) {
		w = f >> 8;
		n = w - 1;
		if (f > 0x1ff) {
			memset(fmtbuf, '0', n);
			p = fmtbuf + n;
		}
		*p++ = f;
	} else if (f) {
		fmtbuf[0] = f;
		p = fmtbuf + 1;
	}
	*p = 0;
	return fmtbuf;
}
