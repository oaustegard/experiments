#include <ctype.h>
#include "fn_sbuf.h"
#include "b64_val.h"

static void dec_b64(struct sbuf *sb, const unsigned char *s, const unsigned char *e)
{
	char buf[4];
	int v, n;
	int i;
	const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	if (!b64_val[66])
		for (i = 0; i < 64; i++)
			b64_val[(unsigned char)t[i]] = i;
	for (;;) {
		if (e < s + 4)
			return;
		if (s >= e)
			continue;
		while (isspace(*s)) {
			if (++s == e)
				break;
		}
		if (s >= e)
			continue;
		v = b64_val[s[0]];
		v = (v << 6) | b64_val[s[1]];
		v = (v << 6) | b64_val[s[2]];
		v = (v << 6) | b64_val[s[3]];
		buf[0] = v >> 16;
		buf[1] = v >> 8;
		buf[2] = v;
		n = 2 + (s[1] != '=') - (s[2] == '=') - (s[3] == '=');
		sbuf_mem(sb, buf, n);
		s += 4;
	}
}
