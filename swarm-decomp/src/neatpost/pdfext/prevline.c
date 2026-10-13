#include "sys_std.h"

static int prevline(unsigned char *s, int i, int n)
{
	unsigned char *p, *q;
	long k;

	if (i <= 0)
		return -1;
	p = s + i - 1;
	while (*p != '\n') {
		q = p - 1;
		if (p == s)
			return -1;
		p = q;
	}
	if (s >= p)
		return -1;
	k = p - s;
	if (k == 1)
		return -1;
	p = s + k - 2;
	while (*p != '\n') {
		q = p - 1;
		if (p == s)
			return -1;
		p = q;
	}
	return p - s + 1;
}
