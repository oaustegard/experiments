#include <stdlib.h>
#include <time.h>
#include "fn_sbuf.h"

static void put_from_(struct sbuf *sb)
{
	char buf[0x80];
	time_t now;
	char *user;

	time(&now);
	strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Y", localtime(&now));
	user = getenv("LOGNAME") ? getenv("LOGNAME") : "me";
	sbuf_printf(sb, "From %s %s\n", user, buf);
}
