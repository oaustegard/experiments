#include "tr_args.h"
#include "ext.h"
#include "fn_sbuf_append.h"
#include "fn_sbuf_done.h"
#include "fn_sbuf_empty.h"
#include "fn_sbuf_init.h"
#include "fn_sbuf_len.h"
#include "fn_str_get.h"
#include "fn_str_set.h"

static void tr_chop(struct tr_args *a)
{
	int id = map(a->arg[0]);
	struct sbuf b;

	if (str_get(id)) {
		sbuf_init(&b);
		sbuf_append(&b, str_get(id));
		if (!sbuf_empty(&b)) {
			sbuf_cut(&b, sbuf_len(&b) - 1);
			str_set(id, sbuf_buf(&b));
		}
		sbuf_done(&b);
	}
}
