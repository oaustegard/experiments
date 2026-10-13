#include <ctype.h>
#include "hyph_ext.h"
#include "fn_memset.h"
#include "hcode_mapchar.h"

static char *hcode_mapchar(char *s)
{
	int n = dict_get(hcodedict, s);

	if (n >= 0)
		return strcpy(s, hcodedst[n]);
	if (s[1])
		return s;
	*s = isalpha((unsigned char)*s) ? tolower((unsigned char)*s) : '.';
	return s;
}
