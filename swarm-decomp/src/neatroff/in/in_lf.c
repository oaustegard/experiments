#include "vars.h"
#include "fn_in_lf.h"
#include "fn_memset.h"

void
in_lf(char *s, int n)
{
	struct src *p = buf;

	while (p != NULL && p->fp == NULL)
		p = p->prev;
	if (s != NULL)
		snprintf((char *)p, 1024, "%s", s);
	p->lnum = n - 1;
}
