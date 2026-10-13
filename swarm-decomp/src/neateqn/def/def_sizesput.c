#include <stdio.h>
#include <string.h>

void def_sizesput(const char *name, const char **out)
{
	int i, k;

	for (i = 0; i < 32; i++)
		if (strcmp(bracketsizes[i][0], name) == 0)
			break;
	if (i == 32) {
		for (i = 0; i < 32; i++)
			if (bracketsizes[i][0][0] == 0)
				break;
		if (i == 32)
			return;
	}
	snprintf(bracketsizes[i][0], 64, "%s", name);
	for (k = 0; k < 7; k++)
		snprintf(bracketsizes[i][k + 1], 64, "%s", out[k] ? out[k] : "");
}
