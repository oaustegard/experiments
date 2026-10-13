#include <string.h>
#include "fn_sbuf.h"

static void toutf8(const char *cs, struct sbuf *sb, const char *s, int n)
{
	char **tbl;
	const char *t;
	unsigned char c;
	int i;
	int w, m;

	if (!cs) {
		sbuf_mem(sb, s, n);
		return;
	}
	w = strcmp("windows-1256", cs);
	m = strcmp("iso-8859-6", cs);
	if (w) {
		if (m) {
			sbuf_mem(sb, s, n);
			return;
		}
		tbl = iso8859_6;
	} else {
		tbl = !m ? iso8859_6 : cp1256;
	}
	for (i = 0; i < n; i++) {
		c = s[i];
		t = tbl[c];
		if (t)
			sbuf_str(sb, t);
		else
			sbuf_chr(sb, c);
	}
}
