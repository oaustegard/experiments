#include "mkmsg.h"

static struct mkmsg **put_msg(struct mkmsg **out, struct mkmsg *m) {
	struct mkmsg *c = m->child;

	*out++ = m;
	while (c) {
		out = put_msg(out, c);
		c = c->next;
	}
	return out;
}
