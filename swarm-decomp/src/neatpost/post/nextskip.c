#include <ctype.h>

static void nextskip(void)
{
	int c;

	do
		c = getc(stdin);
	while (isspace(c));
	ungetc(c, stdin);
}
