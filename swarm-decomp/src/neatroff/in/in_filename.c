#include "vars.h"
#include "fn_in_filename.h"

char *
in_filename(void)
{
	struct src *p = buf;

	if (p == NULL)
		return "-";
	while (p->fp == NULL) {
		p = p->prev;
		if (p == NULL)
			return "-";
	}
	if (*(char *)p == 0)
		return "-";
	return (char *)p;
}
