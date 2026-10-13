#include "fn_sbuf.h"

static void dec_qp(struct sbuf *sb, const char *s, const char *e, int flag)
{
	unsigned char c;
	const char *p;
	char d;
	int v;

	while (s < e) {
		c = *s;
		if (c == '=') {
			d = s[1];
			p = s + 2;
			if (d == '\n') {
				s = p;
				continue;
			}
			if (p >= e) {
				sbuf_chr(sb, c);
				s++;
				continue;
			}
			v = (hexval(d) << 4) | hexval(s[2]);
			sbuf_chr(sb, v);
			s += 3;
		} else {
			if (!(flag & 1) || c != '_')
				sbuf_chr(sb, c);
			else
				sbuf_chr(sb, ' ');
			s++;
		}
	}
}
