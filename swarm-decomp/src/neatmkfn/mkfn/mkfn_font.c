#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
mkfn_font(char *s)
{
	static int idx;
	char *f;

	idx++;
	f = mkfn_subfont;
	if (!f)
		return idx == 1;
	if (strcmp("list", f) == 0) {
		puts(s);
		f = mkfn_subfont;
	}
	if (*f && isdigit((unsigned char)*f) && idx == (int)strtol(f, NULL, 10))
		return 1;
	return strcmp(f, s) == 0;
}
