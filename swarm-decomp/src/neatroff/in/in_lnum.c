#include "vars.h"
#include "fn_in_lnum.h"

int
in_lnum(void)
{
	struct src *p = buf;

	while (p != NULL && p->fp == NULL)
		p = p->prev;
	return p != NULL ? p->lnum : 0;
}
