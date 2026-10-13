#include "fn_sbuf.h"

char *base64(const unsigned char *s, int n)
{
	struct sbuf *sb = sbuf_make();
	int i;
	for (i = 0; n > i; i += 3) {
		unsigned int v = s[i], b1 = 0, b2 = 0;
		char c2 = '=', c3 = '=';
		if (n > i + 1) {
			b1 = s[i + 1] << 8;
		}
		if (n > i + 2)
			b2 = s[i + 2];
		v = (v << 16) | b1 | b2;
		if (i && i % 57 == 0)
			sbuf_printf(sb, "\n");
		sbuf_printf(sb, "%c", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v >> 18]);
		sbuf_printf(sb, "%c", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(v >> 12) & 0x3f]);
		if (n > i + 1)
			c2 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(v >> 6) & 0x3f];
		sbuf_printf(sb, "%c", c2);
		if (n > i + 2)
			c3 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[v & 0x3f];
		sbuf_printf(sb, "%c", c3);
	}
	sbuf_printf(sb, "\n");
	return sbuf_done(sb);
}
