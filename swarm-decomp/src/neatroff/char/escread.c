#include <string.h>

#include "c_ec.h"
#include "fn_nreg.h"
#include "fn_escread.h"
#include "utf8read.h"

int escread(char **p, char **t)
{
	static char buf[4096];
	char *s, *d;
	int kind;
	unsigned char c, c1;
	int r;

	if (**p == 0)
		return -1;
	*t = buf;
	utf8read(p, buf);
	r = 0;
	if (buf[0] == c_ec) {
		utf8read(p, buf + 1);
		if (buf[1] == '(') {
			utf8read(p, buf);
			utf8read(p, buf + strlen(buf));
			return 0;
		}
		if (*nreg('C') == 0 && buf[1] == '[') {
			s = *p;
			d = buf;
			if (*s != 0 && *s != ']') {
				do {
					*p = s + 1;
					*d++ = *s;
					s = *p;
				} while (*s != 0 && *s != ']');
			}
			*d = 0;
			s = *p;
			if (*s == ']')
				*p = s + 1;
			return 0;
		}
		kind = buf[1];
		if (strchr("CDfhmsvXx<>", kind)) {
			buf[0] = 0;
			if (strchr("*fgkmns", kind)) {
				s = *p;
				if (*s == '(') {
					d = buf;
					*d++ = s[1];
					*d++ = s[2];
					s += 3;
				} else if (*nreg('C') == 0 && *s == '[') {
					d = buf;
					c = s[1];
					s++;
					while (c && c != ']') {
						*d++ = c;
						c = *++s;
					}
					if (c == ']')
						s++;
				} else {
					d = buf;
					*d++ = *s;
					s++;
				}
				*d = 0;
				*p = s;
			}
			if (strchr("bCDhHjlLNoRSvwxXZ?", kind)) {
				c1 = **p;
				d = buf;
				s = *p + 1;
				c = *s;
				while (c != 0 && c != c1) {
					*d++ = c;
					c = *++s;
				}
				if (c == c1)
					s++;
				*d = 0;
				*p = s;
			}
			if (kind != 'C')
				r = kind;
		}
		return r;
	}
	if (buf[0] == 4) {
		utf8read(p, buf + 1);
	}
	return 0;
}
