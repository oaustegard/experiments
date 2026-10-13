#include "tr_args.h"
#include "ext.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_init.h"
#include "fn_str_set.h"
#include "libc.h"

static void tr_coi(struct tr_args *a)
{
	char *s = a->arg[0];
	char *f = a->arg[1];
	FILE *fp;
	struct sbuf sb;
	char buf[1024];
	char *t;

	if (!s || !*s || !f || !*f)
		return;
	fp = fopen(f, "r");
	if (!fp)
		return;
	sbuf_init(&sb);
	while (fgets(buf, 1024, fp))
		sbuf_append(&sb, buf);
	t = sbuf_buf(&sb);
	str_set(map(s), t);
	sbuf_done(&sb);
	fclose(fp);
}
