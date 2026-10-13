#include <stdlib.h>
#include <string.h>
#include "macrobody.h"

static void macrobody(struct sbuf *b, char *term)
{
	int c, d, nl = 1;
	char *s;

	in_back('\n');
	cp_copymode(1);
	for (;;) {
		c = cp_next();
		if (c < 0) {
			s = NULL;
			break;
		}
		if (b && !nl)
			sbuf_add(b, c);
		nl = 0;
		if (c == '\n') {
			d = cp_next();
			if (d != c_cc) {
				in_back(d);
				nl = 0;
				continue;
			}
			s = read_name(*nreg('C'));
			if (strcmp(term, s) == 0) {
				in_push(term, 0);
				in_back(c_cc);
				break;
			}
			if (b) {
				sbuf_add(b, c_cc);
				sbuf_append(b, s);
			}
			free(s);
			nl = 0;
		}
	}
	free(s);
	cp_copymode(0);
}
