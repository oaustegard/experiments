#include <ctype.h>
#include <string.h>
#include "brk_class.h"

static int brk_match(const char *s, int c, int flags)
{
	char first = *s;
	int res = first == '^';
	const char *p, *beg;
	const struct brk_class *cl;
	unsigned char c0;
	int lo, hi;

	p = s;
	if (first == '^')
		p++;
	beg = p;
	c0 = *p;
	if ((flags & 4) && c <= 0x7f && isupper(c))
		c = tolower(c);
	p = beg;
	if (!c0)
		goto end;
	for (;;) {
		if (c0 == '[' && p[1] == ':') {
			const char *q = p + 1;
			for (cl = brk_classes; cl != brk_classes + 11; cl++)
				if (strncmp(cl->name, q, strlen(cl->name)) == 0 &&
				    brk_match(cl->chars, c, flags) == 0)
					return res;
			p += brk_len(p);
		} else {
			lo = uc_dec(p);
			p += uc_len(p);
			hi = lo;
			if (*p == '-' && p[1] != ']' && p[1]) {
				p++;
				hi = uc_dec(p);
				p += uc_len(p);
			}
			if (flags & 4) {
				if (lo <= 0x7f && isupper(lo))
					lo = tolower(lo);
				if (hi <= 0x7f && isupper(hi))
					hi = tolower(hi);
			}
			if (c >= lo && c <= hi)
				return res;
		}
		c0 = *p;
		if (!c0 || (p != beg && c0 == ']'))
			break;
	}
end:
	res = first != '^';
	return res;
}
