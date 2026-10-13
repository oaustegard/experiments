#include <ctype.h>
#include "fn_errmsg.h"
#include "fn_dev_mnt.h"

int dev_pos(char *s)
{
	int i;

	if (isdigit(*s)) {
		i = strtol(s, NULL, 10);
		if ((unsigned)i <= 31 && fn_font[i] != NULL)
			return i;
		errmsg("neatroff: bad font position %s\n", s);
		return -1;
	}
	for (i = 1; i < 32; i++)
		if (strcmp(fn_name[i], s) == 0)
			return i;
	if (strcmp(fn_name[0], s) == 0)
		return 0;
	return dev_mnt(0, s, s);
}
