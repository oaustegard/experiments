#include <stdarg.h>
#include "libc.h"
#include "sbuf.h"
#include "fn_sbuf_append.h"

void sbuf_printf(struct sbuf *b, const char *fmt, ...) {
	va_list ap;
	char buf[1024];

	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	sbuf_append(b, buf);
}
