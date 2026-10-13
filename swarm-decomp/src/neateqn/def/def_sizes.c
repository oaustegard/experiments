#include <string.h>

void def_sizes(const char *name, const char **out)
{
	int i, k;
	const char **p;
	const char (*r)[64];

	for (i = 0; i != 32; i++)
		if (strcmp(bracketsizes[i][0], name) == 0)
			break;
	if (i == 32) {
		*out = name;
		return;
	}
	*out = name;
	p = out;
	r = bracketsizes[i] + 1;
	for (k = 1; k < 8; k++, r++)
		*p++ = (*r)[0] ? *r : 0;
}
