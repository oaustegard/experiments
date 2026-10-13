#include "libc.h"
#include "fn_errdie.h"

void errdie(const char *s)
{
	fputs(s, stderr);
	exit(1);
}
