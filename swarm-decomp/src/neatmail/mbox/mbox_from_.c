#include <string.h>
static char *mbox_from_(char *p, char *end)
{
	char *q;
	while (p && p + 6 < end) {
		if (*p == 'F' && p[1] == 'r' && p[2] == 'o' && p[3] == 'm' && p[4] == ' ')
			return p;
		q = memchr(p, '\n', end - p);
		if (!q)
			return 0;
		if (q + 7 >= end)
			return 0;
		p = q + 1;
		continue;
	}
	return 0;
}
