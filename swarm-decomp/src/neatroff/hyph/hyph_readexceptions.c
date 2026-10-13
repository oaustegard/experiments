#include <ctype.h>
#include "hw_add.h"

static void hyph_readexceptions(long s0)
{
	char buf[256], *p;
	char *s = (char *)s0;

	while (*s) {
		p = buf;
		while (*s && !isspace((unsigned char)*s))
			*p++ = *s++;
		*p = 0;
		hw_add(buf);
		while (*s && isspace((unsigned char)*s))
			s++;
	}
}
