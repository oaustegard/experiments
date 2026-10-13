#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int msg_set(char *s, size_t n, char **out, long *outlen, char *key, char *val) {
	char *p = msg_get(s, n, key);
	char *end = s + n;
	char *last;
	char *cur;
	int hl = 0;
	int vl;
	int pre;
	long total;
	char *buf;

	if (p) {
		hl = hdrlen(p, end - p);
		vl = strlen(val);
	} else {
		cur = s;
		do {
			last = cur;
			cur = hdrnext(cur, end);
		} while (cur);
		vl = strlen(val);
		p = last + hdrlen(last, end - last);
	}
	pre = p - s;
	long rest = n - hl;
	total = vl + rest;
	*outlen = total;
	buf = malloc(total + 1);
	*out = buf;
	if (!buf)
		return 1;
	memcpy(buf, s, pre);
	memcpy(buf + pre, val, vl);
	memcpy(buf + vl + pre, s + pre + hl, (long)n - pre - hl + 1);
	buf[total] = 0;
	return 0;
}
