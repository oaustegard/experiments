#include <string.h>
#include "gtype.h"

int def_type(const char *name)
{
	int i;
	char **q;

	for (i = 0; i != 128 && gtypes[i].name[0] != 0; i++)
		if (strcmp(name, gtypes[i].name) == 0)
			goto found;
	for (i = 0; i != 5; i++)
		if (strcmp(name, puncs[i]) == 0)
			return 0x70;
	for (q = binops; q != binops + 27; q++)
		if (strcmp(name, *q) == 0)
			return 0x30;
	for (q = relops; q != relops + 41; q++)
		if (strcmp(name, *q) == 0)
			return 0x40;
	for (q = bracketleft; q != bracketleft + 6; q++)
		if (strcmp(name, *q) == 0)
			return 0x50;
	for (q = bracketright; q != bracketright + 6; q++)
		if (strcmp(name, *q) != 0)
			continue;
		else
			return 0x60;
	return -1;
found:
	return gtypes[i].val;
}
