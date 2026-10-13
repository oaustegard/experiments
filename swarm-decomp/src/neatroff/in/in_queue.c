#include "fn_in_queue.h"

void
in_queue(char *s)
{
	if (nfiles > 15)
		return;
	if (s == NULL)
		s = "";
	snprintf(files[nfiles++], 1024, "%s", s);
}
