#include "cmap.h"
#include "cmap_dst.h"
#include "ext.h"

char *cmap_map(char *s)
{
	int n = dict_get(cmap, s);

	if (n < 0)
		return s;
	return cmap_dst[n];
}
