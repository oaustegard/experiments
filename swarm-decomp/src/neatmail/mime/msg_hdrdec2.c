#include <ctype.h>
#include <string.h>
#include "fn_sbuf.h"
#include "toutf8.h"
#include "dec_qp.h"

static void msg_hdrdec2(struct sbuf *sb, const char *p, const char *e)
{
	const char *a, *b, *d;
	struct sbuf *sb2;
	char cs[128];
	unsigned char c;
	unsigned int len;
	int enc;

	while (p < e) {
		const char *q;
		c = *p;
		q = p + 1;
		if (c == '=' && *q == '?' &&
		    (a = memchr(p + 2, '?', e - q)) != NULL &&
		    (b = memchr(a + 1, '?', e - a)) != NULL &&
		    (d = memchr(b + 1, '?', e - b)) != NULL &&
		    d[1] == '=') {
			enc = tolower((unsigned char)a[1]);
			sb2 = sbuf_make();
			len = a - p - 2;
			if (len > 127)
				len = 127;
			memcpy(cs, p + 2, len);
			cs[len] = 0;
			if (enc == 'b')
				dec_b64(sb2, b + 1, d);
			else
				dec_qp(sb2, b + 1, d, 1);
			p = d + 2;
			while (isspace((unsigned char)*p)) {
				if (p + 1 >= e)
					break;
				p++;
			}
			sbuf_rep(sb2, '\n', '\t');
			len = sbuf_len(sb2);
			toutf8(cs, sb, sbuf_buf(sb2), len);
			sbuf_free(sb2);
		} else {
			sbuf_chr(sb, c);
			p = q;
		}
	}
}
