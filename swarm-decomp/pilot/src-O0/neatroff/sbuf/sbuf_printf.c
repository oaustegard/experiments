#include <stdarg.h>
#include <stdio.h>

void sbuf_printf(struct sbuf *sb, const char *fmt, ...)
{
	char buf[0x400];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	sbuf_append(sb, buf);
}
