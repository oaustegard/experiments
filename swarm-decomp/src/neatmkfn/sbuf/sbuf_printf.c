#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "sbuf.h"
void sbuf_mem(struct sbuf *sb, const void *p, int n);

void sbuf_printf(struct sbuf *sb, const char *fmt, ...)
{
	char buf[256];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	sbuf_mem(sb, buf, strlen(buf));
}
