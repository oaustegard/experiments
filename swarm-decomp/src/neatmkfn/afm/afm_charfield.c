#include <ctype.h>

static char *afm_charfield(char *s, char *d)
{
	char *p = s;
	char c = *p;

	while (c != 0) {
		if (isspace(c))
			break;
		if (c == ';')
			break;
		*d++ = c;
		c = *++p;
	}
	for (;;) {
		while (isspace(*p))
			p++;
		if (*p != ';')
			break;
		p++;
	}
	*d = 0;
	return p;
}
