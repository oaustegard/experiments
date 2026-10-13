#include "fn_charnext.h"
#include "fn_sstr_next.h"
#include "fn_sstr_back.h"
#include "sstr.h"

int charread(char **p1, char *p2) {
	int n = sstr_n;
	char *s = *p1;
	int r;
	sstr_n = n + 1;
	sstr_bufs[n] = sstr_s;
	sstr_s = s;
	r = charnext(p2, sstr_next, sstr_back);
	n = sstr_n - 1;
	*p1 = sstr_s;
	sstr_n = n;
	sstr_s = sstr_bufs[n];
	return r;
}
