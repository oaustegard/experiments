#include <stdlib.h>
#include "fn_pdf_ws.h"
#include "fn_pdf_len.h"
#include "fn_pdf_dval_val.h"
#include "fn_sbuf_mem.h"
#include "pdf_strcopy.h"

static int pdf_strcopy(unsigned char *s, int end, int i, struct sbuf *sb)
{
	int v, len, p, k;
	unsigned char c;

	v = pdf_dval_val(s, end, i, "/Length");
	if (v < 0)
		return 0;
	len = strtol(s + v, NULL, 10);
	i += pdf_len(s, end, i);
	p = i + pdf_ws(s, end, i);
	if (end <= p + len + 14)
		return 0;
	k = p + 6;
	if (s[k] == '\r')
		k = p + 7;
	k += len + 1;
	c = s[k];
	if (c == '\r' || c == ' ') {
		k++;
		c = s[k];
	}
	return sbuf_mem(sb, (char *)s + p, k + 10 + (c == '\n') - p);
}
