#include "hyph_ext.h"
#include "fn_memset.h"
#include "hw_add.h"

static void hw_add(char *s)
{
	int len = strlen(s) + 1;
	int n = hw_n;
	int k, c;
	long off;
	char *w, *h, *p;

	if (n == 0x4000)
		return;
	off = hwword_len;
	if ((unsigned)(off + len) > 0x40000)
		return;
	w = hwword + off;
	h = memset(hwhyph + off, 0, len);
	c = (unsigned char)*s;
	p = s + 1;
	k = 0;
	while (c) {
		if (c == '-') {
			h[k] = 1;
		} else {
			w[k] = c;
			k++;
		}
		c = (unsigned char)*p++;
	}
	w[k] = 0;
	hwoff[n] = off;
	pref_put(hwpref, w, n);
	hwword_len += k + 1;
	hw_n++;
}
