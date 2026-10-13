#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "sbuf.h"
#include "fn_sbuf_mem.h"

int sbuf_printf(struct sbuf *b, const char *fmt, ...)
{
	va_list ap;
	char tmp[256];

	va_start(ap, fmt);
	vsnprintf(tmp, 256, fmt, ap);
	va_end(ap);
	return sbuf_mem(b, tmp, strlen(tmp));
}
