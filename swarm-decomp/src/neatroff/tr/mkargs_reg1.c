#include "sbuf.h"
#include "read_name.h"
#include "mkargs_req.h"
#include "ext.h"

static void mkargs_reg1(struct sbuf *b)
{
	char *s = read_name(*nreg('C'));

	sbuf_append(b, s);
	sbuf_add(b, 0);
	free(s);
	mkargs_req(b);
}
