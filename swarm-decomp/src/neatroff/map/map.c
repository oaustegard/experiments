#include "dict.h"
#include "map_dict.h"
#include "fn_errdie.h"

int map(char *s)
{
	int i;

	if (!*s)
		return 0;
	if (*s == '.' && s[1] && !s[2])
		return (unsigned char)s[1];
	if (!mapdict)
		mapdict = dict_make(0x1000, 1);
	i = dict_idx(mapdict, s);
	if (i < 0) {
		dict_put(mapdict, s, 0);
		i = dict_idx(mapdict, s);
		if (i > 0x1eff) {
			errdie("neatroff: increase NREGS\n");
		}
	}
	return i + 0x100;
}
