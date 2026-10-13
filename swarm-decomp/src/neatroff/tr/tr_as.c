#include "tr_args.h"
#include "ext.h"
#include "fn_str_get.h"
#include "fn_str_set.h"
#include "fn_xmalloc.h"
#include "libc.h"

static void tr_as(struct tr_args *a)
{
	int id = map(a->arg[0]);
	char *s = "";
	char *t;
	char *p;
	size_t len = 0;

	if (str_get(id)) {
		s = str_get(id);
		len = strlen(s);
	}
	t = a->arg[1];
	if (!t)
		t = "";
	else
		len += strlen(t);
	p = xmalloc(len + 1);
	strcpy(stpcpy(p, s), t);
	str_set(id, p);
	free(p);
}
