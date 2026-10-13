#include <ctype.h>
#include "sys_std.h"

int pdf_find(unsigned char *s, int end, int a, int b)
{
	int first, count, gen, k, p, r, x;
	unsigned char *q, *pp;

	x = prevline(s, prevline(s, end, a), a);
	if (x < 0)
		return -1;
	if (sscanf((char *)s + x, "%d", &x) != 1)
		return -1;
	if (end <= x)
		return -1;
	if (x < 0)
		return -1;
	q = memchr(s + x, '\n', end - x);
	if (q == 0)
		return -1;
	p = q - s + 1;
	if (p < 0)
		return -1;
	for (;;) {
		if (end <= p)
			return -1;
		r = p;
		if (sscanf((char *)s + r, "%d %d", &first, &count) != 2)
			return -1;
		for (k = 0; k < count; k++) {
			q = memchr(s + r, '\n', end - p);
			if (q == 0)
				return -1;
			r = q - s + 1;
			p = r;
			if (r < 0)
				return -1;
			if (sscanf((char *)s + r, "%d %d", &x, &gen) != 2)
				return -1;
			if (first + k == a && gen == b)
				goto found;
		}
	}
found:
	if (x >= end)
		return -1;
	if (x < 0)
		return -1;
	{
		unsigned char *o = (unsigned char *)strstr((char *)s + x, "obj");
		if (o == 0)
			return -1;
		x = o - s + 3;
	}
	if (end <= x)
		return x;
	pp = s + x;
	while (isspace(*pp)) {
		x++;
		pp++;
		if (end <= x)
			break;
	}
	return x;
}
