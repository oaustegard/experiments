#include "cmap.h"
#include "cmap_dst.h"
#include "cmap_n.h"
#include "cmap_src.h"
#include "ext.h"
#include "libc.h"

void cmap_add(char *s, char *t)
{
	int i = dict_get(cmap, s);
	int n;

	if (i < 0) {
		n = cmap_n;
		if (n <= 511) {
			strcpy(cmap_src[n], s);
			strcpy(cmap_dst[n], t);
			dict_put(cmap, cmap_src[n], n);
			cmap_n++;
		}
	} else
		strcpy(cmap_dst[i], t);
}
