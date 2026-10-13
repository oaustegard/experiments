#include <ctype.h>
#include "ex_arg.h"

static char *ex_arg(char *s, char *buf)
{
	unsigned char c;

	while (isspace(c = *s))
		s++;
	if (c == '"') {
		for (c = *++s; c && c != '"'; c = *s) {
			if (c == '\\' && s[1])
				s++;
			*buf++ = *s++;
		}
		*buf = 0;
		return c ? s + 1 : s;
	} else {
		while (c && !isspace(c)) {
			if (c == '\\' && s[1])
				s++;
			*buf++ = *s++;
			c = *s;
		}
		*buf = 0;
		return s;
	}
}
