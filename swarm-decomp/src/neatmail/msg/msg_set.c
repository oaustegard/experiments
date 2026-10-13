#include <stddef.h>
#include <stdlib.h>
#include <string.h>

char *msg_get(char *s, size_t n, char *key);
int hdrlen(char *s, size_t n);

int msg_set(char *s, size_t n, char **out, long *outlen, char *key, char *val) {
	char *end = s + n;
	char *p = msg_get(s, n, key);
	char *last;
	char *cur;
	int hl;
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
		hl = hdrlen(last, end - last);
		p = last + hl;
		hl = 0;
	}
	pre = p - s;
	total = vl + n - hl;
	*outlen = total;
	buf = malloc(total + 1);
	*out = buf;
	if (!buf)
		return 1;
	memcpy(buf, s, pre);
	memcpy(buf + pre, val, vl);
	memcpy(buf + vl + pre, s + pre + hl, n - pre - hl + 1);
	buf[total] = 0;
	return 0;
}
