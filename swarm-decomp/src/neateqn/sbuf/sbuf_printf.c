#include "sbuf.h"
#include <stdarg.h>
#include <stdio.h>

void
sbuf_printf(struct sbuf *b, const char *fmt, ...)
{
	char buf[1008];
	va_list ap;

	va_start(ap, fmt);
	vsprintf(buf, fmt, ap);
	va_end(ap);
	sbuf_append(b, buf);
}
