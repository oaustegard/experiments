#include <ctype.h>
#include "hy_add.h"

static void hyph_readpatterns(long s0)
{
	char buf[256], *p;
	char *s = (char *)s0;

	while (*s) {
		p = buf;
		while (*s && !isspace((unsigned char)*s))
			*p++ = *s++;
		*p = 0;
		hy_add(buf);
		while (*s && isspace((unsigned char)*s))
			s++;
	}
}
