#include <stdio.h>
#include <string.h>

int
mkfn_script(char *s, int x)
{
	char *p;
	char *t;

	if (!mkfn_scripts) {
		if (x == 1 || !s)
			return 1;
		if (strcmp("DFLT", s) == 0)
			p = mkfn_scripts = "DFLT";
		else
			p = mkfn_scripts = "latn";
	} else {
		p = mkfn_scripts;
		if (strcmp("list", p) == 0) {
			puts(s ? s : "");
			p = mkfn_scripts;
		}
	}
	if ((t = strchr(s, ' ')))
		*t = '\0';
	return strstr(p, s) != 0;
}
