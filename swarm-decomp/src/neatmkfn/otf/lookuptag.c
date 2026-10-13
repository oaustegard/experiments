#include <string.h>

static char *lookuptag(const char *p)
{
	static char tag[16];

	sprintf(tag, "%s:%s", p + 16, p[0] ? p : "DFLT");
	if (p[8])
		sprintf(tag + strlen(tag), ":%s", p + 8);
	return tag;
}
