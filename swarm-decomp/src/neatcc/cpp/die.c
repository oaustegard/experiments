#include "cpp_protos.h"

void die(char *fmt, ...)
{
	char b[512];
	va_list ap;
	va_start(ap, fmt);
	vsprintf(b, fmt, ap);
	va_end(ap);
	write(2, b, strlen(b));
	exit(1);
}
