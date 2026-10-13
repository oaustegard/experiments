#include "libc.h"
#include "fn_map.h"

static void cmddef(char *s, int *nm, char **rest)
{
	char buf[128] = {0};
	char *p;
	long n;
	int m;

	p = strchr(s, '=');
	if (p) {
		n = p - s;
		if (n > 127)
			n = 127;
		memcpy(buf, s, n);
		m = map(buf);
		*rest = p + 1;
		*nm = m;
	} else {
		buf[0] = *s;
		m = map(buf);
		*rest = s + 1;
		*nm = m;
	}
}
