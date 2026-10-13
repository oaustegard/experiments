#include <stdarg.h>
#include "libc.h"

static void outnn(char *fmt, ...) {
	va_list ap;

	va_start(ap, fmt);
	out_nl = strchr(fmt, '\n') != NULL;
	vfprintf(stdout, fmt, ap);
	va_end(ap);
}
