#include <ctype.h>
#include <string.h>

static char *hdrnext(char *p, char *end) {
	char *q;
	unsigned char c;

	if (p + 1 >= end)
		return NULL;
	c = *p;
	for (;;) {
		if (c == '\n')
			return NULL;
		if (c == '\r' && p[1] == '\n')
			return NULL;
		q = memchr(p, '\n', end - p - 1);
		if (!q)
			return NULL;
		if (isalpha((unsigned char)q[1]))
			return q + 1;
		p = q + 1;
		if (p + 1 >= end)
			return NULL;
		c = *p;
	}
}
