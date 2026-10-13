#include <stdio.h>
#include "font.h"

static void tilleol(FILE *fp, char *s)
{
	int c = fgetc(fp);
	while (c != -1 && c != '\n') {
		*s++ = c;
		c = fgetc(fp);
	}
	*s = '\0';
	if (c != -1)
		ungetc('\n', fp);
}
