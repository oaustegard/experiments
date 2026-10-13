#include <stdio.h>
#include <string.h>
#include "ps_state.h"

void outinfo(char *key, char *val)
{
	if (strcmp("Author", key) == 0)
		snprintf(ps_author, 256, "%s", val);
	if (strcmp("Title", key) == 0)
		snprintf(ps_title, 256, "%s", val);
}
