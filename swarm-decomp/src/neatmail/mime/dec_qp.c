#include "fn_sbuf.h"

/* the goto keeps the p >= e literal path in the shared literal tail (gcc layout) */
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
			if (p >= e)
				goto lit;
			v = (hexval(d) << 4) | hexval(s[2]);
			sbuf_chr(sb, v);
			s += 3;
			continue;
		}
lit:
		sbuf_chr(sb, !(flag & 1) || c != '_' ? c : ' ');
		s++;
	}
}
