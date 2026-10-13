#include <stdarg.h>
#include "libc.h"
#include "fn_out.h"

void out(char *fmt, ...) {
	va_list ap;

	if (!out_nl)
		outnn("\n");
	va_start(ap, fmt);
	out_nl = strchr(fmt, '\n') != NULL;
	vfprintf(stdout, fmt, ap);
	va_end(ap);
}
