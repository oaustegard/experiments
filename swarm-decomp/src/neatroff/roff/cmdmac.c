#include <stdio.h>
#include <unistd.h>
#include "fn_in_queue.h"

static int cmdmac(char *dir, char *name)
{
	char buf[1024];
	int r;

	snprintf(buf, sizeof(buf), "%s/%s.tmac", dir, name);
	if (access(buf, 4) != 0)
		snprintf(buf, sizeof(buf), "%s/tmac.%s", dir, name);
	if (access(buf, 4) != 0)
		snprintf(buf, sizeof(buf), "%s/%s", dir, name);
	r = access(buf, 4);
	if (r != 0)
		return 1;
	in_queue(buf);
	return 0;
}
