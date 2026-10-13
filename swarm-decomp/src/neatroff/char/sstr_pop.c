#include "sstr.h"

char *
sstr_pop(void)
{
	char *s = sstr_s;

	sstr_n--;
	sstr_s = sstr_bufs[sstr_n];
	return s;
}
