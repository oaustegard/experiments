#include "fn_hy.h"
#include "hcode_mapchar.h"
#include "hy_cget.h"

static void hcode_strcpy(char *d, char *s, int *offs, int dot)
{
	char buf[32];
	int len = 0, src = 0;

	if (dot) {
		*d = '.';
		len = dot;
	}
	while (s[src]) {
		offs[len] = src;
		src += hy_cget(buf, s + src);
		hcode_mapchar(buf);
		len += hy_cput(d + len, buf);
		if (len > 0xdf)
			break;
	}
	if (dot) {
		d[len++] = '.';
	}
	d[len] = 0;
}
