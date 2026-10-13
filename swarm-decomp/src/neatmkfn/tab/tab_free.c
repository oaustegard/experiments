#include <stdlib.h>
#include "tab.h"

void tab_free(struct tab *t)
{
	free(t->key);
	free(t->val);
	free(t->idx);
	free(t);
}
