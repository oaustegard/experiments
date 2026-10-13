#include <string.h>
#include "fonttype.h"

static int fonttype(char *path)
{
	char *s = strrchr(path, '.');
	int t = '1';

	if (s) {
		t = 't';
		if (strcmp(".ttf", s) != 0 && strcmp(".otf", s) != 0 &&
		    strcmp(".ttc", s) != 0)
			t = strcmp(".otc", s) == 0 ? 't' : '1';
	}
	return t;
}
