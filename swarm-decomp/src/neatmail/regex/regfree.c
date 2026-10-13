#include <stdlib.h>
#include "regex_t.h"

void regfree(regex_t *re)
{
	struct rprog *p = re->prog;
	int i;

	for (i = 0; i < p->n; i++)
		if (!p->slots[i].flag)
			free(p->slots[i].str);
	free(p->slots);
	free(p);
}
