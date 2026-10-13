#include <stdlib.h>
#include <string.h>
#include "sbuf.h"

char *sbuf_buf(struct sbuf *sb)
{
	char *s;
	int n;
	if (sb->s) {
		s = sb->s;
		n = sb->n;
	} else {
		sb->sz = 1;
		s = sb->s = malloc(1);
		n = sb->n;
		if (n) {
			memcpy(s, 0, n);
		} else
			n = 0;
	}
	s[n] = 0;
	return sb->s;
}
