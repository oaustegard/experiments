#include <stddef.h>
#include <string.h>

int hdrlen(char *s, size_t n) {
	char *p = s;
	char *q;
	char c;

	for (;;) {
		q = strchr(p, '\n');
		if (!q) {
			p += strlen(p);
			break;
		}
		p = q + 1;
		c = *p;
		if (!c)
			break;
		if (c != ' ' && c != '\t')
			break;
	}
	return p - s;
}
