#include "vars.h"
#include "fn_in_so.h"
#include "fn_errmsg.h"
#include "fn_memset.h"
#include "fn_xmalloc.h"
#include "libc.h"

void
in_so(char *s)
{
	struct src *p;
	FILE *fp;

	if (s != NULL && *s != '\0') {
		fp = fopen(s, "r");
		if (fp == NULL) {
			errmsg("neatroff: failed to open <%s>\n", s);
			return;
		}
	} else {
		fp = stdin;
		if (fp == NULL) {
			errmsg("neatroff: failed to open <%s>\n", s);
			return;
		}
	}
	p = xmalloc(sizeof *p);
	memset(p, 0, sizeof *p);
	p->prev = buf;
	buf = p;
	p->fp = fp;
	p->lnum = 1;
	if (s != NULL)
		snprintf((char *)p, 1024, "%s", s);
}
