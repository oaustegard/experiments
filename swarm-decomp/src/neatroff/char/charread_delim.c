#include <string.h>

#include "fn_charnext.h"
#include "fn_sstr_next.h"
#include "fn_sstr_back.h"
#include "sstr.h"

int charread_delim(char **p1, char *p2, char *delim)
{
	int n = sstr_n;
	char *s = *p1;
	int r, e;
	sstr_n = n + 1;
	sstr_bufs[n] = sstr_s;
	sstr_s = s;
	r = charnext(p2, sstr_next, sstr_back);
	e = strcmp(p2, delim);
	n = sstr_n - 1;
	if (e == 0)
		r = -1;
	s = sstr_s;
	sstr_n = n;
	sstr_s = sstr_bufs[n];
	*p1 = s;
	return r;
}
