#include <string.h>
#include "gtype.h"

void def_typeput(const char *name, int val)
{
	int i;

	for (i = 0; i != 128 && gtypes[i].name[0] != 0; i++)
		if (strcmp(name, gtypes[i].name) == 0)
			break;
	if (i != 128) {
		strcpy(gtypes[i].name, name);
		gtypes[i].val = val;
	}
}
