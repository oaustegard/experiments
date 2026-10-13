#include "read_until.h"

static void read_until(struct sbuf *b, char *end, int (*next)(), void (*back)())
{
	char buf[32], s[32];
	int c;

	c = next();
	while (c >= 0) {
		if (c != 4) {
			back(c);
			if (c == '\n')
				break;
			if (end) {
				charnext(buf, next, back);
				if (strcmp(end, buf) == 0)
					break;
			} else {
				if (c == ' ' || c == '\t')
					break;
				charnext(buf, next, back);
			}
			charnext_str(s, buf);
			sbuf_append(b, s);
		}
		c = next();
	}
}
