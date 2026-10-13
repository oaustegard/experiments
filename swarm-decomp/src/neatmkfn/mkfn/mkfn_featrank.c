#include <string.h>
#include "scriptorder.h"

int
mkfn_featrank(char *lang, char *feat)
{
	static struct mkfn_scriptent *order;
	struct mkfn_scriptent *p;
	char *f;

	if (order && strcmp(lang, order->name) == 0)
		goto found;
	order = NULL;
	for (p = scriptorder; p != scriptorder + 30; p++)
		if (strcmp(lang, p->name) == 0)
			order = p;
found:
	if (order && (f = strstr(order->feat, feat)))
		return f - order->feat;
	return 1000;
}
