#include <string.h>

static char *hdr_nextfield(const char *s, const char *e)
{
	while (s && s < e && *s != ';') {
		if (*s++ == '"') {
			s = memchr(s, '"', e - s);
			if (s)
				s++;
		}
	}
	if (s && s + 2 < e)
		return s + 1;
	return NULL;
}
