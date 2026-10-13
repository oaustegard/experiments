#include "sstr.h"

void
sstr_push(char *s)
{
	int n = sstr_n;

	sstr_n = n + 1;
	sstr_bufs[n] = sstr_s;
	sstr_s = s;
}
