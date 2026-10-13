#include <ctype.h>
#include <stddef.h>
#include "fn_msg_get.h"

static int msg_stat(char *s, size_t n, int len, int dflt) {
	char *p, *b, *q, *e;
	long l = len;

	p = msg_get(s, n, "status:");
	if (!p)
		return dflt;
	b = p + 7;
	e = s + n;
	q = b + l;
	while (q < e && isspace((unsigned char)*b)) {
		b++;
		q++;
	}
	return isalpha((unsigned char)*q) ? *q : dflt;
}
