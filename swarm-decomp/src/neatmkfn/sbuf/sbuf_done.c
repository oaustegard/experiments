#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

void *sbuf_done(struct sbuf *sb)
{
	char *s = sb->s;
	int n;
	char *r;
	if (!s) {
		sb->sz = 1;
		s = sb->s = malloc(1);
		n = sb->n;
		if (n) {
			memcpy(s, 0, n);
		} else
			n = 0;
	} else
		n = sb->n;
	s[n] = 0;
	r = sb->s;
	free(sb);
	return r;
}
