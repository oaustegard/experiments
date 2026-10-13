#include <stdio.h>
#include <string.h>

int
mkfn_lang(char *s)
{
	char *t;
	int list;

	if (!mkfn_langs)
		return 1;
	list = strcmp("list", mkfn_langs) == 0;
	if (!s) {
		s = "";
		if (!list)
			goto match;
	}
	if (list)
		puts(s);
	if ((t = strchr(s, ' ')))
		*t = '\0';
match:
	return strstr(mkfn_langs, s) != 0;
}
