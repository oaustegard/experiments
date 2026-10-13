#include <time.h>
#include "fn_sbuf.h"

static void put_id(struct sbuf *sb)
{
	char host[32] = "neatmail.host";
	char buf[0x80];
	time_t now;

	time(&now);
	strftime(buf, sizeof(buf), "%Y%d%m%H%M%S", localtime(&now));
	sbuf_printf(sb, "Message-ID: <%s@%s>\n", buf, host);
}
