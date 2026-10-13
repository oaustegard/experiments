#include <string.h>
#include "fn_pdf_ws.h"
#include "fn_pdf_len.h"
#include "pdf_copy.h"

static char *pdf_copy(unsigned char *s, int end, int i)
{
	static char buf[4096];
	int n;

	i += pdf_ws(s, end, i);
	n = pdf_len(s, end, i);
	if ((unsigned)n > 0xfff)
		n = 0xfff;
	memcpy(buf, s + i, n);
	buf[n] = 0;
	return buf;
}
