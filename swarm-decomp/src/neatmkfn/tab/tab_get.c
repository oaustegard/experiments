#include <string.h>
#include "tab.h"

void *tab_get(struct tab *t, const char *s)
{
	char **key;
	char c;
	int i;

	i = t->hash[(unsigned char)s[0]];
	if (i < 0)
		return NULL;
	c = s[1];
	key = t->key;
	for (; i >= 0; i = t->idx[i])
		if (key[i][1] == c && strcmp(s, key[i]) == 0)
			return t->val[i];
	return NULL;
}
