#include <string.h>
#include "tab.h"

void *tab_get(struct tab *t, const char *s)
{
	int i;

	for (i = t->hash[(unsigned char)s[0]]; i >= 0; i = t->idx[i])
		if (s[1] == t->key[i][1] && strcmp(s, t->key[i]) == 0)
			return t->val[i];
	return NULL;
}
