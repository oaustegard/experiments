#include "hyph_ext.h"
#include "hcode_strcpy.h"
#include "hw_lookup.h"

static int hw_lookup(char *w, char *s)
{
	char buf[256] = {0};
	int offs[256] = {0};
	int len = -1;
	int i, j, k;
	long d;
	char c;

	hcode_strcpy(buf, w, offs, 0);
	j = 0;
	while (buf[j] == '.')
		j++;
	i = pref_prefix(hwpref, buf + j, &len);
	if (i < 0)
		return 1;
	d = hwoff[i] - (long)j;
	for (k = j; buf[k]; k++) {
		c = hwhyph[d + k];
		if (c)
			s[offs[k]] = c;
	}
	return 0;
}
