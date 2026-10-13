#include "sbuf.h"
#include "utf8_len.h"

char *pdftext_static(char *s)
{
	static char buf[1024];
	struct sbuf *b;
	char *q, *r;
	unsigned char c;
	unsigned int cp;
	int hi, lo;

	b = sbuf_make();
	c = *s;
	q = s;
	if (c == 0)
		goto plain;
	for (;;) {
		if ((unsigned char)(c - '(') <= 1)
			break;
		if (c & 0x80)
			break;
		c = *++q;
		if (c == 0)
			goto plain;
	}
	sbuf_str(b, "<FEFF");
	for (; *s; s += utf8_len(*s)) {
		c = *s;
		if ((c & 0xc0) != 0xc0) {
			cp = c;
		} else if (!(c & 0x20)) {
			cp = ((c & 0x1f) << 6) | (s[1] & 0x3f);
		} else if (!(c & 0x10)) {
			cp = ((s[1] << 6) & 0xfc0) | ((c & 0xf) << 12) | (s[2] & 0x3f);
			if (!((unsigned)(cp - 0xe000) <= 0x1fff || cp <= 0xd7ff))
				continue;
		} else if (!(c & 8)) {
			cp = ((c & 7) << 18) | ((s[1] << 12) & 0x3f000) | ((s[2] << 6) & 0xfc0) | (s[3] & 0x3f);
			if (!((unsigned)(cp - 0xe000) <= 0x1fff || cp <= 0xd7ff)) {
				if ((unsigned)(cp - 0x10000) > 0xfffff)
					continue;
				hi = 0xd800 + ((cp - 0x10000) >> 10);
				lo = 0xdc00 + (cp & 0x3ff);
				sbuf_printf(b, "%02X%02X", hi >> 8, hi & 0xff);
				sbuf_printf(b, "%02X%02X", lo >> 8, lo & 0xff);
				continue;
			}
		} else {
			cp = c;
		}
		sbuf_printf(b, "%02X%02X", cp >> 8, cp & 0xff);
	}
	c = '>';
	goto tail;
plain:
	sbuf_chr(b, '(');
	sbuf_str(b, s);
	c = ')';
tail:
	sbuf_chr(b, c);
	r = sbuf_done(b);
	snprintf(buf, 0x400, "%s", r);
	free(r);
	return buf;
}
