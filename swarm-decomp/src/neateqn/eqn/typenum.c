#include "libc.h"

static int typenum(const char *s)
{
	if (s[0] == 'o' && s[1] == 'p') {
		if (s[2] == '\0')
			return 0x20;
	}
	if (strcmp(s, "operator") == 0)
		return 0x20;
	if (strcmp(s, "bin") == 0 || strcmp(s, "binary") == 0)
		return 0x30;
	if (strcmp(s, "rel") == 0 || strcmp(s, "relation") == 0)
		return 0x40;
	if (strcmp(s, "open") == 0 || strcmp(s, "opening") == 0)
		return 0x50;
	if (strcmp(s, "close") == 0 || strcmp(s, "closing") == 0)
		return 0x60;
	if (strcmp(s, "punct") == 0 || strcmp(s, "punctuation") == 0)
		return 0x70;
	return strcmp(s, "inner") == 0 ? 0x80 : 0x10;
}
