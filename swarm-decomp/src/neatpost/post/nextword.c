#include <ctype.h>

static void nextword(char *s)
{
	int c;

	nextskip();
	c = getc(stdin);
	while (c >= 0 && !isspace(c)) {
		*s++ = c;
		c = getc(stdin);
	}
	if (c >= 0)
		ungetc(c, stdin);
	*s = 0;
}
