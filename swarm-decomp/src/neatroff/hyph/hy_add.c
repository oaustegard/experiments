#include "fn_memset.h"
#include "hyph_ext.h"
#include "hy_add.h"

static void hy_add(char *s)
{
	int n = hy_n;
	int off, len, k;
	unsigned int c;
	char *p, *d, *q;

	if (n > 0x3fff)
		return;
	off = hypats_len;
	len = strlen(s) + 1;
	if ((unsigned)(off + len) > 0x3ffff)
		return;
	p = hypats + off;
	d = memset(hynums + off, 0, len);
	q = s + 1;
	c = *s;
	k = 0;
	while (c) {
		if ((unsigned)(c - '0') <= 9)
			d[k] = c - '0';
		else
			p[k++] = c;
		c = *q++;
	}
	p[k] = 0;
	hyoff[n] = off;
	pref_put(hypref, p, n);
	hypats_len = off + k + 1;
	hy_n = n + 1;
}
