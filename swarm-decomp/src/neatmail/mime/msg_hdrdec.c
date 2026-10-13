#include <string.h>
#include "fn_sbuf.h"
#include "msg_hdrdec2.h"

char *msg_hdrdec(const char *s)
{
	struct sbuf *sb = sbuf_make();
	msg_hdrdec2(sb, s, s + strlen(s));
	return sbuf_done(sb);
}
