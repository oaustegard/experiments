#include "tr_args.h"
#include "ext.h"
#include "fn_str_get.h"
#include "libc.h"

static void tr_coo(struct tr_args *a)
{
	char *s = a->arg[0];
	char *f = a->arg[1];
	FILE *fp;

	if (!s || !*s || !f || !*f)
		return;
	fp = fopen(f, "w");
	if (!fp)
		return;
	if (str_get(map(s)))
		fputs(str_get(map(s)), fp);
	fclose(fp);
}
