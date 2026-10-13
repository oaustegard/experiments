#include "c_ec.h"
#include "fn_cmap_map.h"
#include "dev_find.h"

void *dev_glyph(char *s, int pos)
{
	int c = *s;
	char *g;

	if ((char)c == 4 || c == c_ec) {
		if (s[1] == c_ec) {
			s++;
			if (s[1] == '(')
				goto two;
		} else if (c == c_ec) {
			if (s[1] == '(') {
two:
				s += 2;
			}
		}
	}
	g = cmap_map(s);
	if (strncmp("GID=", g, 4) == 0)
		return dev_find(g + 4, pos, 1);
	return dev_find(g, pos, 0);
}
